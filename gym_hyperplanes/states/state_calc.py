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

        # each hyperplane described by the angles to each dimension axis of the normal vector plus distance from origin
        self.hyperplane_params_dimension = self.features + 1  # for translation
        # we encode each action as two consequence numbers for up and down direction (both angle and translation)
        # for hyperplane x if we decide to increase angle to the dimension y we get action
        # (self.hyperplane_params_dimension * x + y) while if we decide to decrease it will be
        # (self.hyperplane_params_dimension * x + y + 1)
        # we can decide to increase/decrease angle to axis or move to/from origin
        self.actions_number = self.hyperplane_params_dimension * self.hyperplanes * 2  # for 2 directions
        # overall states depends on number of hyperplane and their dimension (number of features)
        self.states_dimension = self.hyperplanes * self.hyperplane_params_dimension

        self.hyperplane_max_distance_from_origin = self.data_provider.get_max_distance_from_origin()
        self.hyperplane_min_distance_from_origin = self.data_provider.get_min_distance_from_origin()
        self.distance_from_origin_delta_percents = self.data_provider.get_distance_from_origin_delta_percents()

        self.state = np.zeros(self.states_dimension)

        self.best_state = None
        self.best_reward = None

        self.last_areas = None

    def get_state(self):
        return self.state

    def calculate_reward(self):
        areas = dict()

        for ind in range(0, self.data_provider.get_data_size()):
            instance = self.data_provider.get_instance(ind)
            area = ""
            for i in range(0, self.hyperplanes):  #
                hp_features_ind = self.get_hyperplane_features_index(i)
                calc = 0
                for j in range(0, self.features):
                    calc += instance[j] * self.state[hp_features_ind + j]
                area = area + ("0" if calc < self.state[self.get_hyperplane_translation_index(i)] else "1")
            cls = dict() if area not in areas else areas[area]
            # we add all classes we found in the area
            label = instance[len(instance) - 1]
            if label in cls:
                cls[label] = cls[label] + 1
            else:
                cls[label] = 1
            areas[area] = cls

        count = 0
        for key, value in areas.items():
            count -= calculate_miss(value)

        if self.best_reward is None or self.best_reward < count:
            self.best_reward = count
            self.best_state = np.copy(self.state)
            print(self.build_state(self.best_state, 'best state:'))

        self.last_areas = areas
        return count

    def apply_action(self, action):
        action_index = int(action / 2)
        action_direction = UP if action % 2 == 0 else DOWN

        if (action_index + 1) % self.hyperplane_params_dimension == 0:
            self.apply_translation(action_index, action_direction)
        else:
            self.apply_rotation(action_index, action_direction)

    def apply_translation(self, action_index, action_direction):
        if action_direction == UP and self.state[action_index] < self.hyperplane_max_distance_from_origin:
            self.state[action_index] += \
                (self.hyperplane_max_distance_from_origin * self.distance_from_origin_delta_percents) / 100
        if action_direction == DOWN and self.hyperplane_min_distance_from_origin < self.state[action_index]:
            self.state[action_index] -= \
                (self.hyperplane_max_distance_from_origin * self.distance_from_origin_delta_percents) / 100

    def apply_rotation(self, action_index, action_direction):
        new_cos = math.cos(
            (math.acos(self.state[action_index]) + action_direction * math.pi / self.pi_fraction) % math.pi)
        hyperplane_index = self.get_hyperplane_index(action_index)
        hyperplane_features_index = self.get_hyperplane_features_index(hyperplane_index)

        # we need to add/remove this from the 1
        to_remove = 1 - self.calc_sqr_cos_sum(hyperplane_index, new_cos, action_index)
        compensate_direction = 1 if to_remove > 0 else -1
        to_remove = abs(to_remove)

        not_compensate_features = set()
        compensate_delta = to_remove / (self.features - 1 - len(not_compensate_features))
        while (abs(to_remove) > PRECISION) and ((self.features - 1 - len(not_compensate_features)) > 0):
            for i in range(hyperplane_features_index, hyperplane_features_index + self.hyperplane_params_dimension - 1):
                if i not in not_compensate_features:
                    if i != action_index:
                        compensated = self.compensate_other_angles(i, compensate_direction, compensate_delta,
                                                                   1 if self.state[action_index] >= 0 else -1)
                        if compensated != compensate_delta:
                            not_compensate_features.add(i)
                        to_remove -= compensated
                        if to_remove < 0:
                            if abs(to_remove) < PRECISION:
                                break
                            raise Exception(
                                "Got to remove negative value {} for states \n{}".format(to_remove, self.state))
            if (self.features - 1 - len(not_compensate_features)) != 0:
                compensate_delta = to_remove / (self.features - 1 - len(not_compensate_features))
        self.state[action_index] = new_cos
        sum_direction_cosines = self.calc_sqr_cos_sum(hyperplane_index)
        if abs(1 - sum_direction_cosines) > PRECISION:
            raise Exception("Got sum of direction cosines {} for states \n{}".format(sum_direction_cosines, self.state))

    def compensate_other_angles(self, action_index, compensate_direction, compensate_delta, sign):
        compensated = compensate_delta
        res = pow(self.state[action_index], 2) + compensate_direction * compensate_delta
        if compensate_direction > 0:
            if res > 1:
                compensated = compensate_delta - (res - 1)
                self.state[action_index] = 1  # switch quarter if was negative
            else:
                self.state[action_index] = sign * math.sqrt(res)
        else:
            if res < 0:
                compensated = compensate_delta - (0 - res)
                self.state[action_index] = 0
            else:
                self.state[action_index] = sign * math.sqrt(res)
        return compensated

    def get_hyperplane_features_index(self, hyperplane_number):
        return hyperplane_number * (self.features + 1)

    def get_hyperplane_translation_index(self, hyperplane_number):
        return hyperplane_number * (self.features + 1) + self.features

    def get_hyperplane_index(self, action):
        return int(action / self.hyperplane_params_dimension)

    def get_hyperplane_parameter_index(self, action):
        return action % self.hyperplane_params_dimension

    def sample(self):
        selected_action = np.random.randint(0, self.actions_number)
        return selected_action

    def calc_sqr_cos_sum(self, hyperplane_ind, replace_value=None, replace_index=None):
        hp_features_ind = self.get_hyperplane_features_index(hyperplane_ind)
        calc = 0
        for j in range(0, self.features):
            if replace_index is not None and replace_index == hp_features_ind + j:
                calc += pow(replace_value, 2)
            else:
                calc += pow(self.state[hp_features_ind + j], 2)
        return calc

    def reset(self):
        self.state = np.zeros(self.hyperplane_params_dimension * self.hyperplanes)
        for i in range(0, self.hyperplanes):
            self.init_hyperplane(i)
        return self.state

    def init_hyperplane(self, hyperplane_ind):
        each_cos = math.sqrt(1 / (self.hyperplane_params_dimension - 1))
        for i in range(0, self.hyperplane_params_dimension - 1):
            self.state[self.hyperplane_params_dimension * hyperplane_ind + i] = each_cos
        for i in range(0, hyperplane_ind):
            self.apply_translation(
                self.hyperplane_params_dimension * hyperplane_ind + self.hyperplane_params_dimension - 1, UP)

    def print_state(self, best=False):
        print(self.build_state(self.state, 'state:'))
        if best:
            print(self.build_state(self.best_state, 'best state:'))
        print('areas:' + str(self.last_areas))

    def build_state(self, state, name):
        s = name
        delimiter = ''
        for st in state:
            s = s + delimiter + str(round(st, 3))
            delimiter = ' '
        return s
