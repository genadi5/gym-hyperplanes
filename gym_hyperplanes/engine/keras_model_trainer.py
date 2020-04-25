import os
import sys
import time

import gym
import numpy as np

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.callbacks.callbacks import Callback
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

sys.stderr = stderr

import gym_hyperplanes.engine.params as pm

np.random.seed(round(time.time()) % 1000000)


class TargetReachedCallback(Callback):
    def __init__(self, manipulator):
        super(Callback, self).__init__()
        self.manipulator = manipulator

    def on_action_end(self, action, logs={}):
        if self.manipulator.is_done():
            raise KeyboardInterrupt


def execute_hyperplane_search(state_manipulator, config):
    env = gym.make("gym_hyperplanes:hyperplanes-v0")
    env.set_state_manipulator(state_manipulator)

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.get_state().shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(env.get_actions_number()))
    model.add(Activation('softmax'))  # linear

    dqn = DQNAgent(model=model, nb_actions=env.get_actions_number(), nb_steps_warmup=50, policy=BoltzmannQPolicy(),
                   target_model_update=1e-2, memory=SequentialMemory(limit=config.get_max_steps() * 2, window_length=1))
    dqn.compile(Adam(lr=0.1))

    start_time = time.time()
    dqn.fit(env, nb_steps=config.get_max_steps(), verbose=2, nb_max_episode_steps=pm.MAX_EPISODE_STEPS,
            callbacks=[TargetReachedCallback(state_manipulator)])
    print('######### DONE [{}] IN [{}] STEPS WITH REWARD {} WITHIN [{}] SECS!!!!!'.
          format(state_manipulator.get_data_size(), state_manipulator.get_actions_done(),
                 state_manipulator.get_best_reward_ever(), round(time.time() - start_time)))
    return state_manipulator.is_done()
