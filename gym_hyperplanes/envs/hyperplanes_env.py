import gym

from gym_hyperplanes.states.state_calc import StateManipulator


class HyperPlanesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.state_manipulator = None

    def set_state_manipulator(self, state_manipulator=None):
        self.state_manipulator = state_manipulator if state_manipulator is not None else StateManipulator()

    def get_state_shape(self):
        return self.state_manipulator.get_state().shape

    def get_actions_number(self):
        return self.state_manipulator.actions_number

    def print_state(self, title):
        self.state_manipulator.print_state(title)

    def step(self, action):
        self.state_manipulator.apply_action(action)
        reward = self.state_manipulator.calculate_reward()
        return self.state_manipulator.get_state(), reward, reward == 0, {}

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
