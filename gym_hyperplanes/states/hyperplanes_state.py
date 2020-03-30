import operator
import pickle


def save_hyperplanes_state(hyperplane_states, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(hyperplane_states, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_hyperplanes_state(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def create_area_constraints(class_area, hp_state, hp_dist):
    constraints = []
    for i in range(hp_state.shape[1]):
        sign = '>' if class_area & 1 << i > 0 else '<'
        constraints.append(HyperplaneConstraint(hp_state[:, i], hp_dist[i], sign))
    return constraints


class HyperplanesBoundary:
    def __init__(self, hp_state, hp_dist, class_area):
        self.hp_state = hp_state
        self.hp_dist = hp_dist
        self.class_area = class_area

    def get_hp_state(self):
        return self.hp_state

    def get_hp_dist(self):
        return self.hp_dist

    def get_class_area(self):
        return self.class_area


class HyperplanesState:
    def __init__(self, hp_state, hp_dist, areas_to_classes, reward, features_minimums, features_maximums):
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
        self.features_minimums = features_minimums
        self.features_maximums = features_maximums

        self.boundaries = []

    def set_boundaries(self, boundaries):
        self.boundaries = boundaries

    def get_hp_state(self):
        return self.hp_state

    def get_hp_dist(self):
        return self.hp_dist

    def get_areas(self):
        return self.areas_to_classes

    def get_reward(self):
        return self.reward

    def is_supported_class(self, cls):
        return cls in self.classes_to_areas

    def get_class_adapter(self):
        for cls, _ in self.classes_to_areas.items():
            return int if 'int' in str(type(cls)) else str

    def get_class_constraint(self, cls):
        if cls not in self.classes_to_areas:
            return []
        class_areas = self.classes_to_areas[cls]

        constraint_sets = []
        for class_area in class_areas:
            cs = create_area_constraints(class_area, self.hp_state, self.hp_dist)
            constraint_sets.append(HyperplaneConstraintSet(cs))

        for boundary in self.boundaries:
            cs = create_area_constraints(boundary.class_area, self.hp_state, self.hp_dist)
            constraint_sets.append(HyperplaneConstraintSet(cs))
        return constraint_sets

    def get_features_minimums(self):
        return self.features_minimums

    def get_features_maximums(self):
        return self.features_maximums


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
