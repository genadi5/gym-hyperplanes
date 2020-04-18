import logging
import math
import time

import gym_hyperplanes.engine.params as pm
import numpy as np
import numpy_indexed as npi
# import time
from gym_hyperplanes.states.hyperplanes_state import HyperplanesState
from gym_hyperplanes.states.missed_areas import MissedAreas

UP = 1
DOWN = -1
PRECISION = 0.000001
MAX_HYPERPLANES = 20


def print_state(best_hp_state, best_hp_dist, best_areas, best_reward, title):
    logging.debug('+++ {}*******************'.format(title))
    logging.debug('+++++++++++++++++++++++++++++++++++++++++++')
    logging.debug(best_hp_state)
    logging.debug('+++++++++++++++++++++++++++++++++++++++++++')
    logging.debug(best_hp_dist)
    logging.debug('+++++++++++++++++++++++++++++++++++++++++++')
    logging.debug(best_areas)
    logging.debug('+++++++++++++++++++++++++++++++++++++++++++')
    logging.debug(best_reward)
    logging.debug('**********************************')


def print_area(area, area_data, title):
    logging.debug('***{}***area {} ******'.format(title, area))
    logging.debug(area_data)
    logging.debug('**********************************')


def make_area(array, powers):
    return np.bitwise_or.reduce(powers[array > 0])


def calculate_areas(indices, sides, powers, areas, data_provider):
    # all these indices are of same side so let's take example
    unique_side = sides[indices[0]]
    # get all classes of these indices
    classes = np.take(data_provider.get_labels(), indices)
    probabilities = np.take(data_provider.get_labels_probabilities(), indices)
    # get indices by classes
    labels_indices = npi.group_by(classes).split(np.arange(len(classes)))
    # and count how much of of each class we have
    classes_count = {classes[labels_ind[0]]: np.take(probabilities, labels_ind)
                     for labels_ind in labels_indices}

    area = np.bitwise_or.reduce(powers[unique_side])
    areas[area] = classes_count


def calculate_area_accuracy(area_classes):
    sm = 0
    mx = 0
    for key, val in area_classes.items():
        count = len(val)
        sm += count
        if count > mx:
            mx = count
    return sm, mx, math.ceil((mx * 100) / sm)


class StateManipulator:
    def __init__(self, data_provider, hyperplane_config):
        """
        If we have X features then for each hyper plane we have (X + 1) * 2 actions we can take
        X + 1 - for X angles on each dimension/features plus 1 for to/from origin
        * 2 since it can be done in two directions (per action)
        """
        # all features are supposed to be not negative. If they were negative we assume pre-processing were we
        # change (move) origin in order to make all features to be positive

        self.data_provider = data_provider
        self.hyperplane_config = hyperplane_config

        # self.features = 18  # 3 actions per player, two player: 3 * 3 * 2
        self.features = self.data_provider.get_features_size()
        self.hyperplanes = self.hyperplane_config.get_hyperplanes()

        # how much we increase selected angle (to selected dimension)
        self.pi_fraction = self.hyperplane_config.get_rotation_fraction()

        self.area_accuracy = self.hyperplane_config.get_area_accuracy()

        # we encode each action as two consequence numbers for up and down direction (both angle and translation)
        # for hyperplane x if we decide to increase angle to the dimension y we get action
        # (self.hyperplane_params_dimension * x + y) while if we decide to decrease it will be
        # (self.hyperplane_params_dimension * x + y + 1)
        # we can decide to increase/decrease angle to axis or move to/from origin
        # so actions are encoded as follows:
        # |                 hyperplane 0             |                 hyperplane 1             |..| hyperplane n-1
        # |r0_up,r0_down,..,rm_up,rm_down,t_up,t_down,r0_up,r0_down,..,rm_up,rm_down,t_up,t_down,..,r0_up,r0_down,..
        self.actions_number = (self.features + 1) * self.hyperplanes * 2  # for 2 directions
        # overall states depends on number of hyperplane and their dimension (number of features)

        self.hp_max_distance_from_origin = self.data_provider.get_max_distance_from_origin()
        self.hp_min_distance_from_origin = self.data_provider.get_min_distance_from_origin()
        self.dist_from_origin_delta_percents = self.hyperplane_config.get_from_origin_delta_percents()

        # alls hyperplane positions are concatenated in one array
        self.state = np.zeros(self.hyperplanes * (self.features + 1))
        # each hyperplane will be encoded in dedicated column
        # number of rows is like number of features
        self.hp_state = np.zeros([self.features, self.hyperplanes])
        # the 'right' part of hyperplane equality
        self.hp_dist = np.zeros([self.hyperplanes])

        self.powers = np.array([pow(2, i) for i in range(self.hyperplanes)])

        # we always store best results
        self.best_hp_state = None
        self.best_hp_dist = None
        self.best_state = None
        self.best_areas = None
        self.best_reward = None

        self.best_restart_ever = None
        self.best_hp_state_ever = None
        self.best_hp_dist_ever = None
        self.best_state_ever = None
        self.best_areas_ever = None
        self.best_reward_ever = None

        self.total_areas = 0

        self.restarts = -1
        self.actions_done = 0
        self.rotations_done = 0
        self.translations_done = 1  # we assume we have done on translation (on init)
        self.thousand_time = time.time()
        self.thousand_took = 0
        self.start_time = time.time()
        self.reset_stats()

    def get_hp_state(self, complete):
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print_state(self.best_hp_state_ever, self.best_hp_dist_ever, self.best_areas_ever, self.best_reward_ever,
                        'Start state')
        copy_best_areas = dict(self.best_areas_ever)
        missed_areas = {}
        areas_features_bounds = {}
        signs = np.dot(self.data_provider.get_only_data(), self.best_hp_state_ever) - self.best_hp_dist_ever
        areas = np.apply_along_axis(make_area, 1, signs, self.powers)
        for area, value in self.best_areas_ever.items():
            _, _, accuracy = calculate_area_accuracy(value)
            area_data, area_features_minimums, area_features_maximums = self.data_provider.get_area_data(areas == area)
            areas_features_bounds[area] = [area_features_minimums, area_features_maximums]
            if not complete:
                if accuracy < self.area_accuracy:
                    missed_areas[area] = area_data
                    copy_best_areas.pop(area, None)
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        print_area(area, area_data, 'extracting {}'.format(accuracy))
                elif logging.getLogger().isEnabledFor(logging.DEBUG):
                    area_data = self.data_provider.get_area_data(areas == area)
                    print_area(area, area_data, 'preserving with {}'.format(accuracy))
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                area_data = self.data_provider.get_area_data(areas == area)
                print_area(area, area_data, 'preserving with {}'.format(accuracy))

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug('+++ Finished state')
        return MissedAreas(missed_areas, self.best_hp_state_ever, self.best_hp_dist_ever), \
               None if len(copy_best_areas) == 0 else HyperplanesState(self.best_hp_state_ever,
                                                                       self.best_hp_dist_ever, copy_best_areas,
                                                                       self.best_reward_ever,
                                                                       areas_features_bounds,
                                                                       self.data_provider.get_features_minimums(),
                                                                       self.data_provider.get_features_maximums()
                                                                       )

    def calculate_reward(self):
        areas, reward = self._calculate_ratio()
        if self.actions_done % 1000 == 0:
            self.thousand_took = round(time.time() - self.thousand_time)
            self.thousand_time = time.time()
        if self.best_reward is None or self.best_reward < reward:
            self.best_reward = reward
            self.best_hp_state = np.copy(self.hp_state)
            self.best_hp_dist = np.copy(self.hp_dist)
            self.best_state = np.copy(self.state)
            self.best_areas = areas
            self.print_state('Best reward [{}]'.format(self.best_reward))

            if self.best_reward_ever is None or self.best_reward_ever < self.best_reward:
                best_reward_ever_before = self.best_reward_ever
                self.best_reward_ever = self.best_reward
                self.best_restart_ever = self.restarts

                self.best_hp_state_ever = self.best_hp_state
                self.best_hp_dist_ever = self.best_hp_dist
                self.best_state_ever = self.best_state
                self.best_areas_ever = self.best_areas
                self.print_state('!!! Best reward ever [{}] from [{}] in episode {}'.
                                 format(self.best_reward_ever, best_reward_ever_before, self.restarts))
        elif self.actions_done % 1000 == 0:
            self.print_state('steps [1000/{}] in [{}] secs, current reward [{}], total time [{}] in [{}] restarts'.
                             format(self.actions_done, self.thousand_took, reward,
                                    round(time.time() - self.start_time), self.restarts))

        return reward

    def _calculate_ratio(self):
        areas = dict()
        # calculate value of instances per hyperplane
        signs = np.dot(self.data_provider.get_only_data(), self.hp_state) - self.hp_dist
        sides = (signs > 0)

        indices_list = npi.group_by(sides).split(np.arange(len(sides)))
        for indices in indices_list:
            calculate_areas(indices, sides, self.powers, areas, self.data_provider)

        # self.total_areas += (round(time.time()) - start_areas)
        reward = 0
        for key, value in areas.items():
            sm, mx, curr_area_accuracy = calculate_area_accuracy(value)
            reward += sm if curr_area_accuracy >= self.area_accuracy else 0
            # reward -= 0 if curr_area_accuracy >= self.area_accuracy else sm - mx
        return areas, reward

    def apply_action(self, action):
        self.actions_done += 1
        hyperplane = int(action / ((self.features + 1) * 2))
        start_hyperplane = hyperplane * (self.features + 1) * 2

        action_index = int((action - start_hyperplane) / 2)
        action_direction = UP if action % 2 == 0 else DOWN

        if (action_index > 0) and (((action_index + 1) % (self.features + 1)) == 0):
            self.apply_translation(hyperplane, action_direction)
            self.translations_done += 1
        else:
            feature_ind = self.get_feature_index(action_index)
            self.apply_rotation(hyperplane, feature_ind, action_direction)
            self.rotations_done += 1
        self.copy_state()

    def apply_translation(self, hp_ind, action_direction):
        if action_direction == UP and self.hp_dist[hp_ind] < self.hp_max_distance_from_origin:
            self.hp_dist[hp_ind] += (self.hp_max_distance_from_origin * self.dist_from_origin_delta_percents) / 100
        if action_direction == DOWN and self.hp_min_distance_from_origin < self.hp_dist[hp_ind]:
            self.hp_dist[hp_ind] -= (self.hp_max_distance_from_origin * self.dist_from_origin_delta_percents) / 100

    def apply_rotation(self, hp_ind, feature_ind, action_direction):
        new_cos = math.cos(
            (math.acos(self.hp_state[feature_ind, hp_ind]) + action_direction * math.pi / self.pi_fraction) % math.pi)

        # we need to add/remove this from the 1
        to_remove = 1 - self.calc_sqr_cos_sum(hp_ind, new_cos, feature_ind)
        compensate_direction = 1 if to_remove > 0 else -1
        to_remove = abs(to_remove)

        not_compensate_features = set()
        compensate_delta = to_remove / (self.features - 1 - len(not_compensate_features))
        while (abs(to_remove) > PRECISION) and ((self.features - 1 - len(not_compensate_features)) > 0):
            for i in range(self.features):
                if i not in not_compensate_features:
                    if i != feature_ind:
                        compensated = self.compensate_other_angles(i, hp_ind, compensate_direction, compensate_delta,
                                                                   1 if self.hp_state[feature_ind, hp_ind] >= 0 else -1)
                        if compensated != compensate_delta:
                            not_compensate_features.add(i)
                        to_remove -= compensated
                        if to_remove < 0:
                            if abs(to_remove) < PRECISION:
                                break
                            raise Exception(
                                "Got to remove negative value {} for states \n{}".format(to_remove, self.hp_state))
            if (self.features - 1 - len(not_compensate_features)) != 0:
                compensate_delta = to_remove / (self.features - 1 - len(not_compensate_features))
        self.hp_state[feature_ind, hp_ind] = new_cos
        sum_direction_cosines = self.calc_sqr_cos_sum(hp_ind)
        if abs(1 - sum_direction_cosines) > PRECISION:
            raise Exception(
                "Got sum of direction cosines {} for states \n{}".format(sum_direction_cosines, self.hp_state))

    def compensate_other_angles(self, feature_ind, hp_ind, compensate_direction, compensate_delta,
                                sign):
        compensated = compensate_delta
        res = pow(self.hp_state[feature_ind, hp_ind], 2) + compensate_direction * compensate_delta
        if compensate_direction > 0:
            if res > 1:
                compensated = compensate_delta - (res - 1)
                self.hp_state[feature_ind, hp_ind] = 1  # switch quarter if was negative
            else:
                self.hp_state[feature_ind, hp_ind] = sign * math.sqrt(res)
        else:
            if res < 0:
                compensated = compensate_delta - (0 - res)
                self.hp_state[feature_ind, hp_ind] = 0
            else:
                self.hp_state[feature_ind, hp_ind] = sign * math.sqrt(res)
        return compensated

    def get_feature_index(self, action):
        return action % self.features

    def sample(self):
        if (self.rotations_done / self.translations_done) < pm.ROTATION_TRANSLATION_MAX_RATIO:
            action = np.random.randint(0, self.actions_number)
        else:
            # print('Not enough translations {} while rotations {} ration ro be up to {}'.
            #       format(self.translations_done, self.rotations_done, pm.ROTATION_TRANSLATION_MAX_RATIO))
            hyperplane = np.random.randint(0, self.hyperplanes)
            start_hyperplane = hyperplane * (self.features + 1) * 2
            start_hyperplane_translation = start_hyperplane + self.features * 2
            action = start_hyperplane_translation + round(time.time()) % 2
        return action

    def calc_sqr_cos_sum(self, hp_ind, replace_value=None, feature_ind=None):
        res = sum(pow(self.hp_state[:, hp_ind], 2))
        if feature_ind is not None:
            res = res - pow(self.hp_state[feature_ind, hp_ind], 2) + pow(replace_value, 2)
        return res

    def copy_state(self):
        for i in range(self.hyperplanes):
            start_hp_features = i * (self.features + 1)
            self.state[start_hp_features:start_hp_features + self.features] = self.hp_state[:, i]
            self.state[start_hp_features + self.features] = self.hp_dist[i]

    def add_hyperplane(self):
        if MAX_HYPERPLANES <= self.hyperplanes:
            return False
        self.hyperplanes += 1
        self.actions_number = (self.features + 1) * self.hyperplanes * 2

        self.hp_state = self.best_hp_state
        self.hp_dist = self.best_hp_dist

        self.hp_state = np.hstack((self.hp_state, np.zeros((self.features, 1))))
        self.hp_state[:, -1] = math.sqrt(1 / self.features)

        self.hp_dist = np.hstack((self.hp_dist, [np.median(self.hp_dist)]))
        self.state = np.hstack((self.state, [0] * (self.features + 1)))
        self.copy_state()

        self.best_hp_state = np.copy(self.hp_state)
        self.best_hp_dist = np.copy(self.hp_dist)
        self.best_state = np.copy(self.state)
        return True

    def reset_stats(self):
        self.restarts += 1
        self.actions_done = 0
        self.rotations_done = 0
        self.translations_done = 1  # we assume we have done on translation (on init)
        self.thousand_time = time.time()
        self.thousand_took = 0
        self.start_time = time.time()

    def reset(self):
        self.reset_stats()

        self.best_hp_state = None
        self.best_hp_dist = None
        self.best_state = None
        self.best_areas = None
        self.best_reward = None

        self.state = np.zeros(self.hyperplanes * (self.features + 1))
        self.hp_state = np.zeros([self.features, self.hyperplanes])
        self.hp_dist = np.zeros([self.hyperplanes])

        self.hp_state[:, :] = math.sqrt(1 / self.features)

        times_each_hyperplane_to_translate = math.floor(
            (100 / (self.hyperplanes + 1)) / self.dist_from_origin_delta_percents)

        for i in range(0, self.hyperplanes):
            for j in range((i + 1) * times_each_hyperplane_to_translate):
                self.apply_translation(i, UP)
        self.copy_state()

        return self.state

    def print_state(self, title):
        sm = '+++++++++++++++++++++++++++++++++++++ {} +++++++++++++++++++++++++++++++++++++'.format(title)
        logging.debug(sm)
        print(sm)
        # print(self.build_state(self.best_state, 'best state:'))
        logging.debug('best areas:' + str(self.best_areas))
        em = 'best reward[{}],best reward ever[{}],restart[{}],data[{}],steps[{}],restart[{}]:'. \
            format(self.best_reward, self.best_reward_ever, self.best_restart_ever,
                   self.get_data_size(), self.actions_done, self.restarts)
        logging.debug(em)
        print(em)
        eem = '************************************************************************************************'
        logging.debug(eem)
        print(eem)

    def build_state(self, state, name):
        s = name
        delimiter = ''
        for st in state:
            s = s + delimiter + str(round(st, 3))
            delimiter = ' '
        return s

    def get_state(self):
        return self.state

    def get_best_reward_ever(self):
        return self.best_reward_ever

    def is_done(self):
        return self.best_reward_ever == self.get_data_size()
        # return self.best_reward_ever == 0

    def get_data_size(self):
        return self.data_provider.get_actual_data_size()

    def get_actions_done(self):
        return self.actions_done
