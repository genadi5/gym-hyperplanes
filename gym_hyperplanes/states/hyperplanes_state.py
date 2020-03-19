import pickle


def save_hyperplanes_state(hyperplane_states, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(hyperplane_states, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_hyperplanes_state(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


class HyperplanesState:
    def __init__(self, hp_state, hp_dist, areas, reward):
        self.hp_state = hp_state
        self.hp_dist = hp_dist
        self.areas = areas
        self.reward = reward

    def get_hp_state(self):
        return self.hp_state

    def get_hp_dist(self):
        return self.hp_dist

    def get_areas(self):
        return self.areas

    def get_reward(self):
        return self.reward
