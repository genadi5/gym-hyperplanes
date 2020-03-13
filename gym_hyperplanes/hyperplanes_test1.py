import datetime
import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from gym_hyperplanes.iris.iris_data_provider import IrisDataProvider
from gym_hyperplanes.states.state_calc import StateManipulator


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.get_state_shape()
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.get_actions_number()))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    np.random.seed(123)

    env = gym.make("gym_hyperplanes:hyperplanes-v0")
    env.set_state_manipulator(StateManipulator(IrisDataProvider()))
    # env.set_state_manipulator()

    episod_len = 1000000

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    done = False
    start = datetime.datetime.now()
    cur_state = env.reset().reshape(1, env.get_state_shape()[0])
    best_reward = None
    worst_reward = None
    step = 0
    for step in range(episod_len):
        action = dqn_agent.act(cur_state)
        new_state, reward, done, _ = env.step(action)
        if best_reward is None or best_reward < reward:
            best_reward = reward
        if worst_reward is None or worst_reward > reward:
            worst_reward = reward

        # reward = reward if not done else -20
        new_state = new_state.reshape(1, env.get_state_shape()[0])
        dqn_agent.remember(cur_state, action, reward, new_state, done)

        dqn_agent.replay()  # internally iterates default (prediction) model
        dqn_agent.target_train()  # iterates target model

        cur_state = new_state
        if done:
            break

        if step > 0 and step % 100 == 0:
            print("rewards: best {}, worst {} in step {}".format(best_reward, worst_reward, step))
            env.print_state()

    stop = datetime.datetime.now()
    print("rewards: best {}, worst {} in {}".format(best_reward, worst_reward, (stop - start)))
    if not done:
        print("last state in step {}".format(step))
        env.print_state()
        if step % 10 == 0:
            dqn_agent.save_model("trial-{}.model".format(stop))
    else:
        print("success state of step {}".format(step))
        env.print_state()
        dqn_agent.save_model("success.model")
    print("last state")
    env.print_state(best=True)


if __name__ == "__main__":
    main()
