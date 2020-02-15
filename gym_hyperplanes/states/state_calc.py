import math

import numpy as np

UP = 1
DOWN = -1
START_CIRCLE = 0
FULL_CIRCLE = 360


class StateManipulator:
    def __init__(self):
        '''
        If we have X features then for each hyper plane we have (X + 1) * 2 actions we can take
        X + 1 - for X angles on each dimension/features plus 1 for to/from origin
        * 2 since it can be done in two directions (per action)
        '''
        # self.features = 18  # 3 actions per player, two player: 3 * 3 * 2
        self.features = 2
        self.hyperplanes = 2
        self.angle_delta = 45

        self.hyperplane_params_dimension = self.features + 1
        self.actions_number = self.hyperplanes * self.hyperplane_params_dimension * 2
        self.states_dimension = self.hyperplanes * self.hyperplane_params_dimension

        self.max_distance_from_origin = 100  # calculate
        self.distance_from_origin_delta_percents = 1

        self.state = np.zeros(self.states_dimension)

        self.xy = [[1, 1], [10, 10], [10, 60], [20, 70], [60, 60], [80, 80], [70, 10], [90, 10]]
        self.c = [1, 1, 2, 2, 1, 1, 2, 2]

        self.cos = dict()
        self.cossqr = dict()
        for i in range(0, 360, 45):
            cos = math.cos(i)
            self.cos[i] = cos
            self.cossqr[i] = cos * cos

    def get_state(self):
        return self.state

    def calculate_reward(self):
        features = 2
        areas = dict()
        for i in range(0, int(len(self.state) / features)):  # 2 - two features
            h = str(i)
            for ind in range(0, len(self.c)):
                point = self.xy[ind]
                anglesValSum = 0  # all cosinuses should sum in one
                calc = 0
                for j in range(0, features - 1):
                    cos = self.cos[self.state[i * features + j]]
                    anglesValSum += cos
                    calc += point[j] * cos
                cos = 1 - anglesValSum
                calc += point[features - 1] * cos
                side = "-" + h if calc < self.state[i * features + features - 1] else "+" + h
                cls = set() if side not in areas else areas[side]
                cls.add(self.c[ind])

        count = 0
        for key, value in areas.items():
            if len(value) > 1:
                count -= 1
        return count

    def apply_action(self, action):
        action_index = int(action / 2)
        action_direction = UP if action % 2 == 0 else DOWN

        if self.is_hyperplane_translation(action_index):
            self.apply_translation(action_index, action_direction)
        else:
            result = self.apply_rotation(self.state, action_index, action_direction)
            if result is not None:
                self.state[action_index] = result

    def apply_translation(self, action_index, action_direction):
        if action_direction == UP and self.state[action_index] < self.max_distance_from_origin:
            k = self.state[action_index] + \
                (self.max_distance_from_origin * self.distance_from_origin_delta_percents) / 100
            self.state[action_index] += \
                (self.max_distance_from_origin * self.distance_from_origin_delta_percents) / 100
        if action_direction == DOWN and 0 < self.state[action_index]:
            k = self.state[action_index] - \
                (self.max_distance_from_origin * self.distance_from_origin_delta_percents) / 100
            self.state[action_index] -= \
                (self.max_distance_from_origin * self.distance_from_origin_delta_percents) / 100

    def apply_rotation(self, state, action_index, action_direction):
        hyperplane_ind = int(action_index / self.hyperplane_params_dimension)
        hyperplane_start = hyperplane_ind * self.hyperplane_params_dimension
        result = 0
        for i in range(hyperplane_start, hyperplane_start + self.hyperplane_params_dimension - 1):
            result += self.cossqr[state[i]] if i != action_index \
                else self.cossqr[(state[i] + action_direction * self.angle_delta) % FULL_CIRCLE]
        return None if result > 1 else 0

    def is_hyperplane_translation(self, action_index):
        return (action_index + 1) % self.hyperplane_params_dimension == 0

    def get_hyperplane_index(self, action):
        return action % self.hyperplane_params_dimension

    def get_hyperplane_parameter_index(self, action):
        return action / self.hyperplane_params_dimension

    def sample(self):
        return np.random.randint(0, self.actions_number - 1)

    def reset(self):
        self.state = np.zeros(self.hyperplane_params_dimension * self.hyperplanes)
        return self.state
