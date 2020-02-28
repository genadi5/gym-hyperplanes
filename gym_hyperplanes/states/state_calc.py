import math

import numpy as np

UP = 1
DOWN = -1


class StateManipulator:
    def __init__(self):
        """
        If we have X features then for each hyper plane we have (X + 1) * 2 actions we can take
        X + 1 - for X angles on each dimension/features plus 1 for to/from origin
        * 2 since it can be done in two directions (per action)
        """
        # all features are supposed to be not negative. If they were negative we assume pre-processing were we
        # change (move) origin in order to make all features to be positive

        # self.features = 18  # 3 actions per player, two player: 3 * 3 * 2
        self.features = 2
        self.hyperplanes = 2

        # how much we increase selected angle (to selected dimension)
        self.pi_fraction = 6  # pi / self.pi_fraction

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

        self.max_distance_from_origin = 100  # calculate - actually we should be able to move from - to +
        self.distance_from_origin_delta_percents = 5

        self.state = np.zeros(self.states_dimension)

        # example of input data
        self.xy = [[1, 1], [10, 10], [10, 60], [20, 70], [60, 60], [80, 80], [70, 10], [90, 10]]
        # and its classification
        self.labels = [1, 1, 2, 2, 1, 1, 2, 2]

    def get_state(self):
        return self.state

    def calculate_reward(self):
        areas = dict()
        for i in range(0, self.hyperplanes):  #
            h = str(i)
            hp_features_ind = self.get_hyperplane_features_index(i)
            for ind in range(0, len(self.labels)):
                point = self.xy[ind]
                calc = 0
                for j in range(0, self.features):
                    calc += point[j] * self.state[hp_features_ind + j]
                side = "-" + h if calc < self.state[self.get_hyperplane_translation_index(i)] else "+" + h
                cls = set() if side not in areas else areas[side]
                # we add all classes we found in the area
                cls.add(self.labels[ind])
                areas[side] = cls

        count = 0
        for key, value in areas.items():
            if len(value) > 1:
                # each time some area has mor than 1 class we have mis-classification
                count -= 1
        return count

    def apply_action(self, action):
        action_index = int(action / 2)
        action_direction = UP if action % 2 == 0 else DOWN

        if (action_index + 1) % self.hyperplane_params_dimension == 0:
            self.apply_translation(action_index, action_direction)
        else:
            self.apply_rotation(action_index, action_direction)

    def apply_translation(self, action_index, action_direction):
        if action_direction == UP and self.state[action_index] < self.max_distance_from_origin:
            self.state[action_index] += \
                (self.max_distance_from_origin * self.distance_from_origin_delta_percents) / 100
        if action_direction == DOWN and 0 < self.state[action_index]:
            self.state[action_index] -= \
                (self.max_distance_from_origin * self.distance_from_origin_delta_percents) / 100

    def apply_rotation(self, action_index, action_direction):
        new_cos = math.cos(
            math.acos(self.state[action_index] + action_direction * math.pi / self.pi_fraction) % math.pi)
        to_remove = pow(self.state[action_index], 2) - pow(new_cos, 2)  # we need to add/remove this from the 1
        compensate_direction = 1 if to_remove > 0 else -1
        to_remove = abs(to_remove)

        hyperplane_index = self.get_hyperplane_index(action_index)
        hyperplane_features_index = self.get_hyperplane_features_index(hyperplane_index)

        compensate_features = self.features - 1  # angle for one feature changed, others should compensate for 1
        compensate_delta = math.sqrt(to_remove / compensate_features)
        while (to_remove > 0) and (compensate_features > 0):
            for i in range(hyperplane_features_index, hyperplane_features_index + self.hyperplane_params_dimension - 1):
                if i != action_index:
                    compensated = self.compensate_other_angles(i, compensate_direction, compensate_delta)
                    if compensated != compensate_delta:
                        compensate_features -= 1
                    to_remove -= compensated
                else:
                    self.state[i] = new_cos
            if compensate_features != 0:
                compensate_delta = math.sqrt(to_remove / compensate_features)
        sum_direction_cosines = self.calc_sqr_cos_sum(hyperplane_index)
        if abs(1 - sum_direction_cosines) > 1.001:
            raise "Got sum of compensate_direction cosines {} for states {}".format(sum_direction_cosines, self.state)

    def compensate_other_angles(self, action_index, compensate_direction, compensate_delta):
        compensated = compensate_delta
        if self.state[action_index] > 0:
            if compensate_direction > 0:
                self.state[action_index] += compensate_delta
                if self.state[action_index] > 1:
                    compensated = compensate_delta - (self.state[action_index] - 1)
                    self.state[action_index] = 1
            else:
                self.state[action_index] -= compensate_delta
                if self.state[action_index] < 0:
                    compensated = compensate_delta - (0 - self.state[action_index])
                    self.state[action_index] = 0
        else:
            if compensate_direction > 0:
                self.state[action_index] -= compensate_delta
                if self.state[action_index] < -1:
                    compensated = compensate_delta - (-1 - self.state[action_index])
                    self.state[action_index] = 1  # hyperplane switched
            else:
                self.state[action_index] += compensate_delta
                if self.state[action_index] > 0:
                    compensated = compensate_delta - self.state[action_index]
                    self.state[action_index] = 0

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
        return np.random.randint(0, self.actions_number)

    def calc_sqr_cos_sum(self, hyperplane_ind):
        hp_features_ind = self.get_hyperplane_features_index(hyperplane_ind)
        calc = 0
        for j in range(0, self.features):
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
