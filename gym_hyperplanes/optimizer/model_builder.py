import math

import numpy as np
from gekko import GEKKO

from gym_hyperplanes.classifiers.hyperplanes_classifier import HyperplanesClassifier


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
    print('Objective:[{}]'.format(objective))
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
        print('Eq:[{}]'.format(constraint))

    for var in vars:
        constraint = var >= 0
        m.Equation(constraint)
        print('Eq:[{}]'.format(constraint))


def find_closest_point(point, required_class, hp_states):
    # we are looping in reverse to start from the most deep area since if point is inside deeper (narrow) area
    # it for sure inside shallower (broader) area
    for hp_state in reversed(hp_states):
        if not hp_state.is_supported_class(required_class):
            continue
        # each state id is isolated areas (on it own deep level) so there can't be that same point will
        # fall in several states
        # thus once first classified it to required class  we finished
        classifier = HyperplanesClassifier(hp_state)
        y = classifier.predict(np.array([point]))
        if y[0] is not None: # we found area for point
            if y[0] == required_class: # this is our class!!!
                print('GREAT!!!!!!')
                return point
            # if we reach here this means that we found area but it is of the wrong class
            # we don't continue since on the same levels should not be matching area and on the
            # upper level (shallower/broader/bigger) it is not relevant
            break

    the_closest_point = None
    for hp_state in hp_states:
        constraints_sets = hp_state.get_class_constraint(required_class)
        if len(constraints_sets) > 0:
            features_minimums = hp_state.get_features_minimums()
            features_maximums = hp_state.get_features_maximums()
            results = []

            for i, constraints_set in enumerate(constraints_sets):
                m = GEKKO(remote=False)  # Initialize gekko
                vars = generate_vars_objective(m, features_minimums, features_maximums, point)
                generate_constraints(m, vars, constraints_set.get_constraints())
                m.solve(disp=False)  # Solve
                results.append(([var.value[0] for var in vars], m.options.objfcnval))

            print("Result: " + str(results))
            min_distance = 0
            for result, constraints in zip(results, constraints_sets):
                if the_closest_point is None:
                    the_closest_point = result
                    min_distance = math.sqrt(sum(map(lambda x: x * x, result[0])))
                else:
                    distance = math.sqrt(sum(map(lambda x: x * x, result[0])))
                    if distance < min_distance:
                        min_distance = distance
                        the_closest_point = result
    return the_closest_point
