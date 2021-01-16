import logging
import time
from configparser import ConfigParser

import numpy as np

import gym_hyperplanes.classifiers.classic_classification as ccc
import gym_hyperplanes.optimizer.model_builder as mb
import gym_hyperplanes.optimizer.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.classifiers.hyperplanes_classifier import DeepHyperplanesClassifier
from gym_hyperplanes.states.dataset_provider import DataSetProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig


def load_solution_constraints(instance_dimension, is_int):
    solution_constraints = None
    if pm.CONSTRAINTS_FILE is not None:
        config = ConfigParser()
        config.read(pm.CONSTRAINTS_FILE)
        lower_bounds = None
        upper_bounds = None
        budget_limit = None
        budget_mode = None
        relations_descs = None

        if config.has_option('BUDGET', 'budget_limit'):
            budget_limit = float(config.get('BUDGET', 'budget_limit'))

        if config.has_option('BUDGET', 'budget_mode'):
            budget_mode = int(config.get('BUDGET', 'budget_mode'))
        else:
            budget_mode = mb.BUDGETING_MODE_ABSOLUTE
        if config.has_option('RANGES', 'lower_bounds'):
            lower_bounds = config.get('RANGES', 'lower_bounds')
        if config.has_option('RANGES', 'upper_bounds'):
            upper_bounds = config.get('RANGES', 'upper_bounds')
        if config.has_option('RELATIONS', 'relations'):
            relations_descs = config.get('RELATIONS', 'relations')

        lb = [None] * instance_dimension
        ub = [None] * instance_dimension
        # is_float = ((lower_bounds is not None) and lower_bounds.__contains__('.')) or \
        #            ((upper_bounds is not None) and upper_bounds.__contains__('.'))
        if lower_bounds is not None:
            for i, s in enumerate(lower_bounds.split(',')):
                s = s.strip()
                if len(s) > 0 and s != 'None':
                    lb[i] = int(s) if is_int else float(s)

        if upper_bounds is not None:
            for i, s in enumerate(upper_bounds.split(',')):
                s = s.strip()
                if len(s) > 0 and s != 'None':
                    ub[i] = int(s) if is_int else float(s)

        relations = []
        if relations_descs is not None:
            for relations_descs in relations_descs.split(','):
                relations_descs = relations_descs.strip()
                relations_desc = relations_descs.split(' ')
                relations.append(
                    mb.ConstraintRelation(int(relations_desc[0]), relations_desc[1], int(relations_desc[2])))

        solution_constraints = mb.SolutionConstraint(lb, ub, budget_limit, relations, budget_mode)
    return solution_constraints


def execute():
    hp_states = hs.load_hyperplanes_state(pm.MODEL_FILE)

    is_int = type(hp_states[0].features_minimums[0]) == np.int32

    required_class = hp_states[0].get_class_adapter()(pm.REQUIRED_CLASS)
    instances = pm.INSTANCES

    dataset = None
    if pm.SOURCE_MODEL_FILE is not None:
        dataset = DataSetProvider('optimizer', pm.SOURCE_MODEL_FILE, HyperplaneConfig())

    solution_constraints = load_solution_constraints(len(instances[0]), is_int)

    results = []

    start = time.time()
    for i, instance in enumerate(instances):
        formatted_instance = [round(i) for i in instance] if is_int else [round(i) for i in instance]
        print('>>>>> #{}/#{} at time {} instance {}'.format((i + 1), len(instance), (time.time() - start),
                                                            formatted_instance))
        start_instance = time.time()
        result, constraints = mb.find_closest_point(instance, required_class, hp_states, dataset, solution_constraints)
        formatted_result = None if result is None else [round(i) for i in result[0]] if is_int else [round(i) for i in
                                                                                                   result[0]]
        print('>>>>> #{}/#{} at time {} result   {}'.format((i + 1), len(instances), (time.time() - start), formatted_result))
        if constraints is None:
            logging.info('<<<<< Done in {}, overall {} for instance #{}/#{}:{} is already of class {}'.
                         format((time.time() - start_instance), (time.time() - start), i, len(instances),
                                formatted_instance,
                                required_class))
        else:
            logging.info('<<<<< Done in {}, overall {} for instance #{}/#{} {} closest point to {} in area {}'.
                         format((time.time() - start_instance), (time.time() - start), i, len(instances), instance,
                                formatted_result,
                                constraints.get_class_area()))
        if result is None:
            logging.info('<<<<<<!!!!! No closest point found for instance {} !!!!!!!!!!!!!'.format(instance))
            continue
        results.append(result[0])
    if len(results) > 0:
        if pm.TRAIN_SET is not None:
            print('Starting testing prediction')
            X = np.array(results)
            y = np.array([required_class] * len(results))

            classifier = DeepHyperplanesClassifier(hp_states)
            hp_score = classifier.score(X, y)
            print('Name: [HYPL_CLASSIFIER], Score: [', hp_score, ']')

            ccc.load_and_test(pm.TRAIN_SET, X, y)
            print('Finished testing prediction')
    else:
        print('Finished testing prediction with no results')


def main():
    pm.load_params()
    execute()


if __name__ == "__main__":
    main()
