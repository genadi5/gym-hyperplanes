import math

import numpy as np

from gym_hyperplanes.states.data_provider import TestDataProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig

UP = 1
DOWN = -1
np.random.seed(123)
PRECISION = 0.000001


def calculate_miss(classes):
    sm = sum(classes.values())
    mx = max(classes.values())
    # return int(((sm - mx) * 100) / sm)
    return sm - mx


class StateManipulator:
    def __init__(self, data_provider=TestDataProvider(), hyperplane_config=HyperplaneConfig()):
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

        # we encode each action as two consequence numbers for up and down direction (both angle and translation)
        # for hyperplane x if we decide to increase angle to the dimension y we get action
        # (self.hyperplane_params_dimension * x + y) while if we decide to decrease it will be
        # (self.hyperplane_params_dimension * x + y + 1)
        # we can decide to increase/decrease angle to axis or move to/from origin
        self.actions_number = (self.features + 1) * self.hyperplanes * 2  # for 2 directions
        # overall states depends on number of hyperplane and their dimension (number of features)

        self.hp_max_distance_from_origin = self.data_provider.get_max_distance_from_origin()
        self.hp_min_distance_from_origin = self.data_provider.get_min_distance_from_origin()
        self.dist_from_origin_delta_percents = self.data_provider.get_distance_from_origin_delta_percents()

        self.state = np.zeros(self.hyperplanes * (self.features + 1))
        self.hp_state = np.zeros([self.features, self.hyperplanes])
        self.hp_dist = np.zeros([self.hyperplanes])

        self.best_state = None
        self.best_areas = None
        self.best_reward = None

        self.last_areas = None

    def get_state(self):
        return self.state

    def calculate_reward(self):
        areas = dict()

        calc = np.dot(self.data_provider.get_data(), self.hp_state)
        signs = calc - self.hp_dist
        sides = (signs > 0).astype(int)
        areas_list = [''.join(side.astype(str)) for side in sides]

        for i in range(len(areas_list)):
            cls = dict() if areas_list[i] not in areas else areas[areas_list[i]]
            # we add all classes we found in the area
            label = self.data_provider.get_label(i)
            if label in cls:
                cls[label] = cls[label] + 1
            else:
                cls[label] = 1
            areas[areas_list[i]] = cls

        count = 0
        for key, value in areas.items():
            count -= calculate_miss(value)

        if self.best_reward is None or self.best_reward < count:
            self.best_reward = count
            self.best_state = np.copy(self.state)
            self.best_areas = areas
            self.print_state('Best reward [{}]'.format(self.best_reward))

        self.last_areas = areas
        return count

    def apply_action(self, action):
        action_index = int(action / 2)
        action_direction = UP if action % 2 == 0 else DOWN

        hp_ind = self.get_hp_index(action_index)

        if (action_index > 0) and ((action_index % self.features) == 0):
            self.apply_translation(hp_ind, action_direction)
        else:
            feature_ind = self.get_feature_index(action_index)
            self.apply_rotation(hp_ind, feature_ind, action_direction)
        self.copy_state()

    def copy_state(self):
        for i in range(self.hyperplanes):
            start_hp_features = i * (self.features + 1)
            self.state[start_hp_features:start_hp_features + self.features] = self.hp_state[:, i]
            self.state[start_hp_features + self.features] = self.hp_dist[i]

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

    def get_hp_index(self, action):
        return int(action / (self.features + 1))

    def get_feature_index(self, action):
        return action % self.features

    def sample(self):
        selected_action = np.random.randint(0, self.actions_number)
        return selected_action

    def calc_sqr_cos_sum(self, hp_ind, replace_value=None, feature_ind=None):
        res = sum(pow(self.hp_state[:, hp_ind], 2))
        if feature_ind is not None:
            res = res - pow(self.hp_state[feature_ind, hp_ind], 2) + pow(replace_value, 2)
        return res

    def reset(self):
        self.best_state = None
        self.best_areas = None
        self.best_reward = None
        self.last_areas = None

        self.state = np.zeros(self.hyperplanes * (self.features + 1))
        self.hp_state = np.zeros([self.features, self.hyperplanes])
        self.hp_dist = np.zeros([self.hyperplanes])

        self.hp_state[:, :] = math.sqrt(1 / self.features)
        for i in range(0, self.hyperplanes):
            for j in range(1, i + 1):
                self.apply_translation(i, UP)
        self.copy_state()

        return self.state

    def print_state(self, title):
        print('+++++{}+++++++++++++++++++++++++++'.format(title))
        print('***********************************************')
        print(self.build_state(self.state, 'last state:'))
        print('last areas:' + str(self.last_areas))
        print('------------------------------------------------')
        print(self.build_state(self.best_state, 'best state:'))
        print('best areas:' + str(self.best_areas))
        print('best reward:' + str(self.best_reward))
        print('***********************************************')

    def build_state(self, state, name):
        s = name
        delimiter = ''
        for st in state:
            s = s + delimiter + str(round(st, 3))
            delimiter = ' '
        return s
