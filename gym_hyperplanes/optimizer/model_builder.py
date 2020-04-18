import concurrent.futures
import math
import operator
import os
import sys

import gym_hyperplanes.optimizer.params as pm
import numpy as np
from gekko import GEKKO
from gym_hyperplanes.classifiers.hyperplanes_classifier import DeepHyperplanesClassifier
from gym_hyperplanes.states.state_calc import make_area


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


def stretched(touch_point, dataset_provider, hp_state, powers, class_area):
    signs = np.dot(dataset_provider.get_only_data(), hp_state.get_hp_state()) - hp_state.get_hp_dist()
    areas = np.apply_along_axis(make_area, 1, signs, powers)
    area_data, area_features_minimums, area_features_maximums = dataset_provider.get_area_data(areas == class_area)

    min_distance = 0
    the_closest_point = None
    for i in range(area_data.shape[0]):
        current_point = list(area_data.iloc[i, 0: -1])
        subtractions = map(operator.sub, touch_point, current_point)
        distance = math.sqrt(sum(map(lambda x: x * x, subtractions)))
        if (the_closest_point is None) or (min_distance > distance):
            the_closest_point = current_point
            min_distance = distance

    return calculate_destination(touch_point, the_closest_point, -pm.FEATURE_BOUND_STRETCH_RATIO)


def closest_to_area(point, powers, hp_state, constraints_set, dataset_provider):
    try:
        m = GEKKO(remote=False)  # Initialize gekko
        if pm.FEATURE_BOUND == pm.FEATURE_BOUND_AREA:
            features_bounds = hp_state.get_area_features_bounds(constraints_set.get_class_area())
            vars = generate_vars_objective(m, features_bounds[0], features_bounds[1], point)
        else:
            vars = generate_vars_objective(m, hp_state.get_features_minimums(),
                                           hp_state.get_features_maximums(), point)
        generate_constraints(m, vars, constraints_set.get_constraints())
        sys.stdout = open(os.devnull, "w")
        m.solve(disp=False)  # Solve
        sys.stdout = sys.__stdout__

        touch_point = [var.value[0] for var in vars]

        if pm.FEATURE_BOUND == pm.FEATURE_BOUND_AREA:
            closest_point = touch_point
        elif pm.FEATURE_BOUND == pm.FEATURE_BOUND_FEATURES:
            closest_point = calculate_destination(point, touch_point, pm.PENETRATION_DELTA)
        elif pm.FEATURE_BOUND == pm.FEATURE_BOUND_STRETCH:
            closest_point = stretched(touch_point, dataset_provider, hp_state, powers, constraints_set.get_class_area())
        else:
            print('Oj wej zmir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            closest_point = touch_point

        closest_array = np.dot(np.array(closest_point), hp_state.hp_state) - hp_state.hp_dist
        closest_area = np.bitwise_or.reduce(powers[closest_array > 0])
        if closest_area != constraints_set.class_area:
            return None, None
        return (closest_point, m.options.objfcnval), constraints_set
    except:
        sys.stdout = sys.__stdout__
        print('Optimizer failed {}'.format(sys.exc_info()))
        return None, None


WORKERS = os.cpu_count()
WORKERS = math.ceil(WORKERS * 0.8)

pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)


def find_closest_point(point, required_class, hp_states, dataset_provider=None):
    classifier = DeepHyperplanesClassifier(hp_states)
    y = classifier.predict(np.array([point]), required_class)
    if y[0] is not None:  # we found area for point
        if y[0] == required_class:  # this is our class!!!
            return (point, 0), None

    constraints_set_list = []
    results = []

    closest_points = []
    for h, hp_state in enumerate(hp_states):
        powers = np.array([pow(2, i) for i in range(len(hp_state.hp_dist))])
        constraints_sets = hp_state.get_class_constraint(required_class)
        if len(constraints_sets) > 0:
            for constraints_set in constraints_sets:
                closest_points.append(
                    pool_executor.submit(closest_to_area, point, powers, hp_state, constraints_set, dataset_provider))
                if len(closest_points) % 10 == 0:
                    print('Submitted for search {} areas'.format(len(closest_points)))
    print('Submitted overall for search {} areas'.format(len(closest_points)))

    processed_areas = 0
    for closest_point in concurrent.futures.as_completed(closest_points):
        result = closest_point.result()[0]
        constraints_set = closest_point.result()[1]
        if result is not None:
            results.append(result)
            constraints_set_list.append(constraints_set)
        processed_areas += 1
        if processed_areas % 10 == 0:
            print('Processed {} out of {} areas'.format(processed_areas, len(closest_points)))

    min_distance = 0
    the_closest_point = None
    the_closest_constraints_set = None
    for result, constraints_set in zip(results, constraints_set_list):
        subtractions = map(operator.sub, point, result[0])
        distance = math.sqrt(sum(map(lambda x: x * x, subtractions)))
        if (the_closest_point is None) or (min_distance > distance):
            the_closest_point = result
            min_distance = distance
            the_closest_constraints_set = constraints_set
    print(the_closest_point)
    return the_closest_point, the_closest_constraints_set
