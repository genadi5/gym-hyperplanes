import random
import time
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from gym_hyperplanes.iris.iris_data_provider import IrisDataProvider
from gym_hyperplanes.pendigits.pen_data_provider import PenDataProvider
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
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
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
                q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future * self.gamma
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
    # env.set_state_manipulator(StateManipulator(PenDataProvider()))
    # env.set_state_manipulator()

    episod_len = 1000000

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    done = False
    start = round(time.time())
    last_period = start
    cur_state = env.reset().reshape(1, env.get_state_shape()[0])
    best_reward = None
    worst_reward = None
    step = 1  # we just for sure init it to 1. later on we will do it again --> step = i + 1
    last_step = 1

    total_step_time = 0
    total_act_time = 0
    total_replay_time = 0
    total_train_time = 0
    for i in range(episod_len):
        step = i + 1
        start_act = round(time.time())
        action = dqn_agent.act(cur_state)
        total_act_time += (round(time.time()) - start_act)

        start_step = round(time.time())
        new_state, reward, done, _ = env.step(action)
        total_step_time += (round(time.time()) - start_step)

        if best_reward is None or best_reward < reward:
            best_reward = reward
        if worst_reward is None or worst_reward > reward:
            worst_reward = reward

        # reward = reward if not done else -20
        new_state = new_state.reshape(1, env.get_state_shape()[0])
        dqn_agent.remember(cur_state, action, reward, new_state, done)

        start_replay = round(time.time())
        dqn_agent.replay()  # internally iterates default (prediction) model
        total_replay_time = (round(time.time()) - start_replay)
        start_train = round(time.time())
        dqn_agent.target_train()  # iterates target model
        total_train_time = (round(time.time()) - start_train)

        cur_state = new_state
        if done:
            break

        if step % 10 == 0:
            avrg_act = total_act_time / step
            avrg_step = total_step_time / step
            avrg_replay = total_replay_time / step
            avrg_train = total_train_time / step
            print('{}/{} steps, {} act/step, {} step/step, {} replay/step, {} train/step'.
                  format(step - last_step, step, avrg_act, avrg_step, avrg_replay, avrg_train))
            env.print_state('{}/{} steps in {} secs'.format(step - last_step, step, round(time.time()) - last_period))
            last_period = round(time.time())
            last_step = step

    stop = round(time.time())
    print("rewards: best {}, worst {} in {}".format(best_reward, worst_reward, stop - start))
    if not done:
        print("last state in step {}".format(step))
        if step % 10 == 0:
            dqn_agent.save_model("trial-{}.model".format(stop))
    else:
        print("success state of step {}".format(step))
        dqn_agent.save_model("success.model")
    env.print_state('Finished in [{}] steps in [{}] secs'.format(step, stop - start))


if __name__ == "__main__":
    main()
