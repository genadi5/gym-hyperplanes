import math
import os
import sys

import numpy as np
from gekko import GEKKO

from gym_hyperplanes.classifiers.hyperplanes_classifier import DeepHyperplanesClassifier, HyperplanesClassifier


def generate_vars_objective(m, features_minimums, features_maximums, point):
    vars = []
    objective = None
    for i, min in enumerate(features_minimums):
        var = m.Var(value=min, lb=min, ub=features_maximums[i])
        vars.append(var)
        if objective is None:
            objective = (var - point[i]) * (var - point[i])
        else:
            objective = objective + (var - point[i]) * (var - point[i])
    # print('Objective:[{}]'.format(objective))
    m.Obj(objective)  # Objective
    return vars


def generate_constraints(m, vars, hyperplane_constraints):
    for hyperplane_constraint in hyperplane_constraints:
        constraint = None
        for i, coefficient in enumerate(hyperplane_constraint.get_coefficients()):
            if constraint is None:
                constraint = coefficient * vars[i]
            else:
                constraint = constraint + coefficient * vars[i]
        if hyperplane_constraint.get_sign() == '<':
            constraint = constraint < hyperplane_constraint.get_d()
        else:
            constraint = constraint >= hyperplane_constraint.get_d()
        m.Equation(constraint)
        # print('Eq:[{}]'.format(constraint))

    for var in vars:
        constraint = var >= 0
        m.Equation(constraint)
        # print('Eq:[{}]'.format(constraint))


def calculate_destination(from_point, to_point, delta):
    # r - fraction of distance between from_point and to_point intermediate point
    # in our case r = 1 + delta
    # so equation
    # R = (1 - r) * from_point + r * to_point
    # will look as
    # R = (1 + delta) * to_point - delta * from_point
    return [(1 + delta) * t - delta * f for f, t in zip(from_point, to_point)]


def find_closest_point(point, required_class, hp_states, penetration_delta):
    classifier = DeepHyperplanesClassifier(hp_states)
    y = classifier.predict(np.array([point]), required_class)
    if y[0] is not None:  # we found area for point
        if y[0] == required_class:  # this is our class!!!
            return point, None

    constraints_set_list = []
    results = []

    for h, hp_state in enumerate(hp_states):
        powers = np.array([pow(2, i) for i in range(len(hp_state.hp_dist))])

        constraints_sets = hp_state.get_class_constraint(required_class)
        if len(constraints_sets) > 0:
            features_minimums = hp_state.get_features_minimums()
            features_maximums = hp_state.get_features_maximums()

            for i, constraints_set in enumerate(constraints_sets):
                try:
                    m = GEKKO(remote=False)  # Initialize gekko
                    vars = generate_vars_objective(m, features_minimums, features_maximums, point)
                    generate_constraints(m, vars, constraints_set.get_constraints())
                    sys.stdout = open(os.devnull, "w")
                    m.solve(disp=False)  # Solve
                    sys.stdout = sys.__stdout__
                    touch_point = [var.value[0] for var in vars]
                    closest_point = calculate_destination(point, touch_point, penetration_delta)

                    closest_array = np.dot(np.array(closest_point), hp_state.hp_state) - hp_state.hp_dist
                    closest_area = np.bitwise_or.reduce(powers[closest_array > 0])
                    if closest_area != constraints_set.class_area:
                        continue

                    results.append((closest_point, m.options.objfcnval))
                    constraints_set_list.append(constraints_set)
                except:
                    sys.stdout = sys.__stdout__
    min_distance = 0
    the_closest_point = None
    the_closest_constraints_set = None
    for result, constraints_set in zip(results, constraints_set_list):
        if the_closest_point is None:
            the_closest_point = result
            min_distance = math.sqrt(sum(map(lambda x: x * x, result[0])))
            the_closest_constraints_set = constraints_set
        else:
            distance = 0
            # their squares
            for s, d in zip(point, result[0]):
                distance += pow(d - s, 2)
            distance = math.sqrt(distance)

            if distance < min_distance:
                min_distance = distance
                the_closest_point = result
                the_closest_constraints_set = constraints_set
    print(the_closest_point)
    return the_closest_point[0], the_closest_constraints_set
