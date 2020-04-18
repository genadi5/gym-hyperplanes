import gym
import numpy as np
from gym_hyperplanes.states.dataset_provider import DataSetProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig
from gym_hyperplanes.states.state_calc import StateManipulator
from keras.callbacks.callbacks import Callback
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

MAX_STEPS = 50000


class TargetReachedCallback(Callback):
    def __init__(self, manipulator):
        super(Callback, self).__init__()
        self.manipulator = manipulator

    def on_action_end(self, action, logs={}):
        if self.manipulator.is_done():
            raise KeyboardInterrupt


# from rl.policy import MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy

env = gym.make('gym_hyperplanes:hyperplanes-v0')
config = HyperplaneConfig(area_accuracy=90, hyperplanes=5,
                          pi_fraction=6, from_origin_delta_percents=10,
                          max_steps=2000, max_steps_no_better_reward=500)
data_provider = DataSetProvider("test",
                                '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/iris/iris.data',
                                config, None)
state_manipulator = StateManipulator(data_provider, config)
env.set_state_manipulator(state_manipulator)

np.random.seed(123)
env.seed(123)
nb_actions = env.get_actions_number()

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.get_state().shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

print(model.summary())

memory = SequentialMemory(limit=MAX_STEPS, window_length=1)
# policy = MaxBoltzmannQPolicy()
# policy = BoltzmannGumbelQPolicy()
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=MAX_STEPS, verbose=2, callbacks=[TargetReachedCallback(state_manipulator)])

dqn.save_weights('dqn_{}_weights.h5f'.format('hyperplanes'), overwrite=True)

dqn.test(env, nb_episodes=5)
