import math
import sys

import numpy as np
from gekko import GEKKO

import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.classifiers.hyperplanes_classifier import HyperplanesClassifier


def generate_vars_objective(m, number_of_features, value, lb, ub, point):
    vars = []
    objective = None
    for i in range(number_of_features):
        var = m.Var(value=value, lb=lb, ub=ub)
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


def find_closest_point(point, required_class):
    hp_state = hs.load_hyperplanes_state('/downloads/hyperplanes/IRIS_result.txt')

    classifier = HyperplanesClassifier(hp_state)
    y = classifier.predict(np.array([point]))
    if y[0] == required_class:
        print('GREAT!!!!!!')
        return point
    else:
        the_closest_point = None
        number_of_features = hp_state.number_of_features()
        constraints_sets = hp_state.get_class_constraint(required_class)

        results = []

        for i, constraints_set in enumerate(constraints_sets):
            try:
                m = GEKKO(remote=False)  # Initialize gekko
                vars = generate_vars_objective(m, number_of_features, 0, 0, 8, point)
                generate_constraints(m, vars, constraints_set.get_constraints())
                m.solve(disp=False)  # Solve
                results.append(([var.value[0] for var in vars], m.options.objfcnval))
            except:
                print('Error: ', sys.exc_info()[1])

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


def main():
    required_class = 'Iris-virginica'
    result = find_closest_point([4.4, 2.9, 1.4, 0.2], required_class)
    print(result)


if __name__ == "__main__":
    main()
