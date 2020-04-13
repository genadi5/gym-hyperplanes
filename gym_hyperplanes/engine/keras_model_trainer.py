import time

import gym
import numpy as np
from keras.callbacks.callbacks import Callback
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

np.random.seed(round(time.time()) % 1000000)


class TargetReachedCallback(Callback):
    def __init__(self, manipulator, start_time):
        super(Callback, self).__init__()
        self.manipulator = manipulator
        self.start_time = start_time

    def on_action_end(self, action, logs={}):
        if self.manipulator.get_best_reward() == 0:
            print('@@@@@@@@@@ FINISHED DATA SIZE {} IN {} STEPS WITHIN {} SECS!!!!!'.
                  format(self.manipulator.get_data_size(), self.manipulator.get_actions_done(),
                         (time.time() - self.start_time)))
            raise KeyboardInterrupt


def execute_hyperplane_search(state_manipulator, config):
    env = gym.make("gym_hyperplanes:hyperplanes-v0")
    env.set_state_manipulator(state_manipulator)
    nb_actions = env.get_actions_number()

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.get_state().shape))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=config.get_max_steps(), window_length=1)
    # policy = MaxBoltzmannQPolicy()
    # policy = BoltzmannGumbelQPolicy()
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=config.get_max_steps(), verbose=2, callbacks=[TargetReachedCallback(state_manipulator, time.time())])
    return state_manipulator.get_best_reward() == 0
