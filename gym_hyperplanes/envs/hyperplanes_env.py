import time

import gym

from gym_hyperplanes.states.state_calc import StateManipulator


class HyperPlanesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.state_manipulator = None
        self.actions_done = 0
        self.total_action_time = 0
        self.total_reward_time = 0

    def set_state_manipulator(self, state_manipulator=None):
        self.state_manipulator = state_manipulator if state_manipulator is not None else StateManipulator()

    def get_state_shape(self):
        return self.state_manipulator.get_state().shape

    def get_actions_number(self):
        return self.state_manipulator.actions_number

    def print_state(self, title):
        self.state_manipulator.print_state(title)

    def step(self, action):
        self.actions_done += 1
        start_action = round(time.time())
        self.state_manipulator.apply_action(action)
        self.total_action_time += (round(time.time()) - start_action)
        start_reward = round(time.time())
        reward = self.state_manipulator.calculate_reward()
        self.total_reward_time += (round(time.time()) - start_reward)

        # self.stats()
        return self.state_manipulator.get_state(), reward, reward == 0, {}

    def stats(self):
        if self.actions_done % 1000 == 0:
            avrg_act = self.total_action_time / self.actions_done
            avrg_rwrd = self.total_reward_time / self.actions_done
            print('{} actions, {} average action, {} average reward'.format(self.actions_done, avrg_act, avrg_rwrd))

    def reset(self):
        return self.state_manipulator.reset()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def sample(self):
        return self.state_manipulator.sample()

    def configure(self, key, value):
        pass
