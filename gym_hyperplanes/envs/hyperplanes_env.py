import gym
import numpy as np
from gym.spaces import Discrete, Tuple, Box


class HyperPlanesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        '''
        If we have X features then for each hyper plane we have (X + 1) * 2 actions we van take
        X - for angles on each dimension/features of hyperplane plus 1 for to/from origin
        * 2 since it can be done in two directions (per action)
        '''
        # self.features = 18  # 3 actions per player, two player: 3 * 3 * 2
        self.features = 2  # 3 actions per player, two player: 3 * 3 * 2
        self.hyperplanes = 3
        self.dimension = (self.features + 1) * 2 * self.hyperplanes
        self.max_distance_from_origin = 100
        self.distance_from_origin_delta_percents = 1

        self.angle_delta = 45

        self.action_space = Discrete(self.dimension)

        l_states = []
        for h in range(0, self.hyperplanes):
            for f in range(self.features):
                l_states.append(Discrete(180 - self.angle_delta))
            l_states.append(Discrete(100 / self.distance_from_origin_delta_percents))
        self.observation_space = Tuple(l_states)

        # lows = [0] * (self.features + 1) * self.hyperplanes
        # highs = []
        # for h in range(0, self.hyperplanes):
        #     for f in range(self.features):
        #         highs.append(180 - self.angle_delta)
        #     highs.append(100 / self.distance_from_origin_delta_percents)
        # self.observation_space = Box(lows, highs, dtype=np.float32, shape=(1, (self.features + 1) * self.hyperplanes))

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def configure(self, key, value):
        pass
