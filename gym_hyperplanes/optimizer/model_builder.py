import concurrent.futures
import logging
import math
import operator
import os
import sys

import numpy as np
from gekko import GEKKO

import gym_hyperplanes.optimizer.params as pm
from gym_hyperplanes.classifiers.hyperplanes_classifier import DeepHyperplanesClassifier
from gym_hyperplanes.states.state_calc import make_area

BUDGETING_MODE_ADDITION = 1
BUDGETING_MODE_ABSOLUTE = 2


class SolutionConstraint:
    def __init__(self, lower_bounds, upper_bounds, budget, budget_mode=BUDGETING_MODE_ADDITION):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.budget = budget
        self.budget_mode = budget_mode

    def get_lower_bounds(self):
        return self.lower_bounds

    def get_upper_bounds(self):
        return self.upper_bounds

    def get_budget(self):
        return self.budget

    def get_budget_mode(self):
        return self.budget_mode


def generate_vars_objective(m, features_minimums, features_maximums, point):
    """
    What is our objective?
    We search minimal distance from given point to points inside the area.
    Distance defined as
    sqrt(sum((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2))
    But
    argmin(sqrt(sum((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2)))
    is equal to
    argmin(sum((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2))
    :param m:
    :param features_minimums:
    :param features_maximums:
    :param point:
    :return:
    """
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


def generate_constraints(m, vars, hyperplane_constraints, features_minimums, features_maximums, point, budget, mode):
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

    for i, var in enumerate(vars):
        constraint = var >= features_minimums[i]
        m.Equation(constraint)
        constraint = var <= features_maximums[i]
        m.Equation(constraint)
        # print('Eq:[{}]'.format(constraint)

    if budget is not None:
        constraint = None
        for i, var in enumerate(vars):
            if mode == BUDGETING_MODE_ABSOLUTE:
                if constraint is None:
                    constraint = m.abs(vars[i] - point[i])
                else:
                    constraint = constraint + m.abs(vars[i] - point[i])
            else:
                if constraint is None:
                    constraint = vars[i] - point[i]
                else:
                    constraint = constraint + vars[i] - point[i]
        constraint = constraint <= budget
        m.Equation(constraint)
        # print('Eq:[{}]'.format(constraint))


def calculate_destination(from_point, to_point, delta):
    """
    Finding point between two given point at fraction 'delta' of the whole distance from to_point
    from_point....distance * (1 - delta)...........middle_point.....distance * delta......to_point
    In our case we may search point on the opposite side of to_point so we change the sign of delta

    r - fraction of distance between from_point and to_point intermediate point
    in our case r = 1 + delta
    so equation
    R = (1 - r) * from_point + r * to_point
    will look as
    R = (1 + delta) * to_point - delta * from_point
    :param from_point:
    :param to_point:
    :param delta:
    :return:
    """
    return [(1 + delta) * t - delta * f for f, t in zip(from_point, to_point)]


def stretched(touch_point, dataset_provider, hp_state, powers, class_area):
    """
    One of search modes - once area closest point to the given instance is found we suggest instead of
    it point which lays between this point and the closest among existing instances
    This mode can be used in case performance of original mode (the most close point of area) is not satisfactory
    :param touch_point:
    :param dataset_provider:
    :param hp_state:
    :param powers:
    :param class_area:
    :return:
    """
    # encoding instances areas
    signs = np.dot(dataset_provider.get_only_data(), hp_state.get_hp_state()) - hp_state.get_hp_dist()
    areas = np.apply_along_axis(make_area, 1, signs, powers)
    # getting instances which belong to the area
    area_data, area_features_minimums, area_features_maximums = dataset_provider.get_area_data(areas == class_area)

    # search the closest among all instances of the area
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


def closest_to_area(point, powers, hp_state, constraints_set, dataset_provider, solution_constraint):
    pm.load_params()
    try:
        m = GEKKO(remote=False)  # Initialize gekko

        # we may want to narrow the boundaries of features to make search more close to the real world
        if pm.FEATURE_BOUND == pm.FEATURE_BOUND_AREA:
            features_bounds = hp_state.get_area_features_bounds(constraints_set.get_class_area())
            features_mins, features_maxs = features_bounds[0].copy(), features_bounds[1].copy()
        else:
            features_mins, features_maxs = hp_state.get_features_minimums().copy(), hp_state.get_features_maximums().copy()
        features_mins, features_maxs = rebuild_vars_bounds(features_mins, features_maxs, solution_constraint)
        if features_mins is None:
            return None, None
        vars = generate_vars_objective(m, features_mins, features_maxs, point)

        generate_constraints(m, vars, constraints_set.get_constraints(), features_mins, features_maxs, point,
                             None if solution_constraint is None else solution_constraint.get_budget(),
                             None if solution_constraint is None else solution_constraint.get_budget_mode())
        sys.stdout = open(os.devnull, "w")
        m.solve(disp=False)  # Solve
        sys.stdout = sys.__stdout__

        # this is the point of the area which is the closest to the given one
        touch_point = [var.value[0] for var in vars]

        # we may want to change it slightly according to configuration
        if pm.FEATURE_BOUND == pm.FEATURE_BOUND_AREA:
            closest_point = touch_point
        elif pm.FEATURE_BOUND == pm.FEATURE_BOUND_FEATURES:
            closest_point = calculate_destination(point, touch_point, pm.PENETRATION_DELTA)
        elif pm.FEATURE_BOUND == pm.FEATURE_BOUND_STRETCH:
            closest_point = stretched(touch_point, dataset_provider, hp_state, powers, constraints_set.get_class_area())
        else:
            logging.info('Oj wej zmir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            closest_point = touch_point

        # due to roundings we can be on the opposite side to area - did not reach it although we very close
        # in this case we try to fix this problem
        for i in range(0, 3):
            closest_array = np.dot(np.array(closest_point), hp_state.hp_state) - hp_state.hp_dist
            closest_area = np.bitwise_or.reduce(powers[closest_array > 0])
            if closest_area == constraints_set.class_area:
                logging.info('@@@@@ Closest point {} at  area {} got at attempt {}'.format(closest_point,
                                                                                           constraints_set.class_area,
                                                                                           (i + 1)))
                return (closest_point, m.options.objfcnval), constraints_set
            logging.info('@@@@@ Result point {} is on the opposite side of {}, going to fix it at attempt {}'.
                         format(closest_point, constraints_set.class_area, (i + 1)))
            closest_point = calculate_destination(point, closest_point, pm.PENETRATION_DELTA)
        return None, None
    except:
        sys.stdout = sys.__stdout__
        logging.info('Optimizer failed {}'.format(sys.exc_info()))
        return None, None


def rebuild_vars_bounds(features_minimums, features_maximums, solution_constraint):
    if solution_constraint is not None and solution_constraint.get_lower_bounds() is not None:
        for i, min in enumerate(features_minimums):
            low_bound = solution_constraint.get_lower_bounds()[i]
            if low_bound is not None and low_bound > min:
                features_minimums[i] = low_bound
    if solution_constraint is not None and solution_constraint.get_upper_bounds() is not None:
        for i, max in enumerate(features_maximums):
            high_bound = solution_constraint.get_upper_bounds()[i]
            if high_bound is not None and high_bound < max:
                features_maximums[i] = high_bound

    for i, min in enumerate(features_minimums):
        if min >= features_maximums[i]:
            return None, None
    return features_minimums, features_maximums


WORKERS = os.cpu_count()
# WORKERS = math.ceil(WORKERS * 0.8)

pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)


def find_closest_point(point, required_class, hp_states, dataset_provider=None, solution_constraint=None):
    classifier = DeepHyperplanesClassifier(hp_states)
    # just in case - point can already be of the requested class, let's check this
    y = classifier.predict(np.array([point]), required_class)
    if y[0] is not None:  # we found area for point
        if y[0] == required_class:  # this is our class!!!
            return (point, 0), None

    constraints_set_list = []
    results = []

    # for each hyperplane state we take all its areas
    # and for each area which defined as to be of required class we get it's constraints
    # and search for the closest point to the given point
    closest_points = []
    for h, hp_state in enumerate(hp_states):
        powers = np.array([pow(2, i) for i in range(len(hp_state.hp_dist))])
        constraints_sets = hp_state.get_class_constraint(required_class)
        if len(constraints_sets) > 0:
            for constraints_set in constraints_sets:
                closest_points.append(
                    pool_executor.submit(closest_to_area, point, powers, hp_state, constraints_set,
                                         dataset_provider, solution_constraint))
    print('Submitted overall for search {} areas'.format(len(closest_points)))

    # collecting results of searches executed in processes in parallel
    processed_areas = 0
    for closest_point in concurrent.futures.as_completed(closest_points):
        result = closest_point.result()[0]
        constraints_set = closest_point.result()[1]
        if result is not None:
            results.append(result)
            constraints_set_list.append(constraints_set)
        processed_areas += 1
        if processed_areas % 50 == 0:
            print('Processed {} out of {} areas'.format(processed_areas, len(closest_points)))

    # finding the closest point to the given one
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
    logging.info(the_closest_point)
    return the_closest_point, the_closest_constraints_set
