import operator
import pickle


def save_hyperplanes_state(hyperplane_states, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(hyperplane_states, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_hyperplanes_state(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


class HyperplanesState:
    def __init__(self, hp_state, hp_dist, areas_to_classes, reward):
        self.hp_state = hp_state
        self.hp_dist = hp_dist
        self.areas_to_classes = areas_to_classes
        self.classes_to_areas = dict()
        for area, classes in self.areas_to_classes.items():
            cls = max(classes.items(), key=operator.itemgetter(1))[0]
            class_areas = set() if cls not in self.classes_to_areas else self.classes_to_areas[cls]
            class_areas.add(area)
            self.classes_to_areas[cls] = class_areas
        self.reward = reward

    def get_hp_state(self):
        return self.hp_state

    def get_hp_dist(self):
        return self.hp_dist

    def get_areas(self):
        return self.areas_to_classes

    def get_reward(self):
        return self.reward

    def number_of_features(self):
        return self.hp_state.shape[0]

    def get_class_constraint(self, cls):
        if cls not in self.classes_to_areas:
            return []
        class_areas = self.classes_to_areas[cls]

        constraint_sets = []
        for class_area in class_areas:
            constraints = []
            for i in range(self.hp_state.shape[1]):
                sign = '<' if class_area & 1 << i > 0 else '>'
                constraints.append(HyperplaneConstraint(self.hp_state[:, i], self.hp_dist[i], sign))
            constraint_sets.append(HyperplaneConstraintSet(constraints))
        return constraint_sets


class HyperplaneConstraint:
    def __init__(self, coefficients, d, sign):
        self.coefficients = coefficients
        self.d = d
        self.sign = sign

    def get_coefficients(self):
        return self.coefficients

    def get_d(self):
        return self.d

    def get_sign(self):
        return self.sign


class HyperplaneConstraintSet:
    def __init__(self, constraints):
        self.constraints = constraints

    def get_constraints(self):
        return self.constraints
