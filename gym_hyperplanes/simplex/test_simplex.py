import math

import numpy as np

import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.classifiers.hyperplanes_classifier import HyperplanesClassifier
from gym_hyperplanes.simplex.simplex_new0 import Simplex


def generate_objective(number_of_features):
    objective = ''
    delimiter = ''
    for i in range(number_of_features):
        objective += delimiter + '1x_' + str(i + 1)
        delimiter = ' + '
    return objective


def generate_constraints(hyperplane_constraints):
    constraints = []
    for hyperplane_constraint in hyperplane_constraints:
        constraint = ''
        for i, coefficient in enumerate(hyperplane_constraint.get_coefficients()):
            sign = '-' if coefficient < 0 else '' if i == 0 else '+'
            if i > 0:
                constraint += ' '
            constraint += sign
            constraint += ' ' + str(abs(coefficient)) + 'x_' + str(i + 1)
        constraint += ' ' + hyperplane_constraint.get_sign() + ' ' + str(hyperplane_constraint.get_d())
        constraints.append(constraint)
    return constraints


def restore_point(lp_system):
    return [3, 1]


def test_simplex():
    objective = ('minimize', '1x_1 + 1x_2')
    constraints = ['1x_1 + 0x_2 >= 3', '1x_1 + 0x_2 <= 9', '0x_1 + 1x_2 >= 1']
    lp_system = Simplex(num_vars=2, constraints=constraints, objective_function=objective)
    print(lp_system.solution)
    print(lp_system.optimize_val)


def find_closest_point(point, state_path, required_class):
    hp_state = hs.load_hyperplanes_state('/downloads/hyperplanes/result.txt')
    # TRANSFORM POINT TO ORIGIN
    classifier = HyperplanesClassifier(hp_state)
    y = classifier.predict(np.array([[40, 40]]))
    if y[0] == required_class:
        print('GREAT!!!!!!')
        return point
    else:
        the_closest_point = None
        number_of_features = hp_state.number_of_features()
        constraints_sets = hp_state.get_class_constraint(required_class)
        closest_points = []
        objective = generate_objective(number_of_features)
        objective_min = ('minimize', objective)
        for i, constraints_set in enumerate(constraints_sets):
            constraints = generate_constraints(constraints_set.get_constraints())
            lp_system = Simplex(num_vars=number_of_features, constraints=constraints, objective_function=objective_min)
            closest_point = restore_point(lp_system)
            closest_points.append(closest_point)

        min_distance = 0
        for closest_point, constraints in zip(closest_points, constraints_sets):
            if the_closest_point is None:
                the_closest_point = closest_point
                min_distance = math.sqrt(sum(map(lambda x: x * x, closest_point)))
            else:
                distance = math.sqrt(sum(map(lambda x: x * x, closest_point)))

        # TRANSFORM CLOSEST POINT TO ORIGINAL COORDINATES
        return closest_points


def main():
    required_class = 1
    point = find_closest_point([40, 40], '/downloads/hyperplanes/result.txt', required_class)
    print(point)


if __name__ == "__main__":
    main()
