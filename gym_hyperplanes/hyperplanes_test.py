import gym
import numpy as np
# from rl.agents.dqn import DQNAgent
from gym_hyperplanes.dqn import DQNAgent
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

env = gym.make('gym_hyperplanes:hyperplanes-v0')
# env = gym.make('CartPole-v0')
np.random.seed(123)
env.seed(123)

env.configure(3, 4)

HPs = 2
angles = 4
nb_actions = env.actions_number
# space_shape = env.observation_space.shape
# nb_actions = 4
# space_shape = 100

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.state.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
# model = Sequential()
state_shape = self.env.observation_space.shape
# model.add(Dense(24, input_dim=env.state.shape[0], activation="relu"))
# model.add(Dense(48, activation="relu"))
# model.add(Dense(24, activation="relu"))
# model.add(Dense(nb_actions))
# model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-3))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
