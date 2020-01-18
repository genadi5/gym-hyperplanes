import gym
import numpy as np

from gym_hyperplanes.states.state_calc import StateCalculator

UP = 1
DOWN = -1
FULL_CIRCLE = 360
START_CIRCLE = 0


class HyperPlanesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        '''
        If we have X features then for each hyper plane we have ((X - 1) + 1) * 2 actions we van take
        (X - 1) - for angles on each dimension/features - 1 of hyperplane plus 1 for to/from origin
        * 2 since it can be done in two directions (per action)
        '''
        # self.features = 18  # 3 actions per player, two player: 3 * 3 * 2
        self.features = 2  # 3 actions per player, two player: 3 * 3 * 2
        self.hyperplanes = 2
        self.hyperplane_params_dimension = self.features
        self.actions_number = self.hyperplanes * self.hyperplane_params_dimension * 2
        self.states_dimension = self.hyperplanes * self.hyperplane_params_dimension

        self.max_distance_from_origin = 100  # calculate
        self.distance_from_origin_delta_percents = 1

        self.angle_delta = 45

        self.state = np.zeros(self.states_dimension)

        self.state_calc = StateCalculator()

    def step(self, action):
        action_index = int(action / 2)
        action_direction = UP if action % 2 == 0 else DOWN

        if self.is_hyperplane_translation(action_index):
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
        else:
            self.state[action_index] = (self.state[action_index] + action_direction * self.angle_delta) % FULL_CIRCLE
        return self.state, self.state_calc.calculate_reward(self.state), self.state_calc.is_done(self.state), {}

    def reset(self):
        self.state = np.zeros(self.hyperplane_params_dimension * self.hyperplanes)
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def sample(self):
        return np.random.randint(0, self.actions_number - 1)

    def configure(self, key, value):
        pass

    def get_hyperplane_index(self, action):
        return action % self.hyperplane_params_dimension

    def get_hyperplane_parameter_index(self, action):
        return action / self.hyperplane_params_dimension

    def is_hyperplane_translation(self, action_index):
        return (action_index + 1) % self.hyperplane_params_dimension == 0
