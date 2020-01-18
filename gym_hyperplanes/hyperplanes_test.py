import gym
import numpy as np

env = gym.make('gym_hyperplanes:hyperplanes-v0')
np.random.seed(123)
env.seed(123)
