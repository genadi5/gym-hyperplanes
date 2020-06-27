import logging
import time

import numpy as np

import gym_hyperplanes.classifiers.classic_classification as ccc
import gym_hyperplanes.optimizer.model_builder as mb
import gym_hyperplanes.optimizer.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.classifiers.hyperplanes_classifier import DeepHyperplanesClassifier
from gym_hyperplanes.states.dataset_provider import DataSetProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig


def execute():
    hp_states = hs.load_hyperplanes_state(pm.MODEL_FILE)
    required_class = hp_states[0].get_class_adapter()(pm.REQUIRED_CLASS)
    instances = pm.INSTANCES

    dataset = None
    if pm.SOURCE_MODEL_FILE is not None:
        dataset = DataSetProvider('optimizer', pm.SOURCE_MODEL_FILE, HyperplaneConfig())

    results = []

    start = time.time()
    for i, instance in enumerate(instances):
        print('>>>>> #{}/#{} at time {} instance {}'.format((i + 1), len(instances), (time.time() - start), instance))
        start_instance = time.time()
        result, constraints = mb.find_closest_point(instance, required_class, hp_states, dataset)
        print('>>>>> #{}/#{} at time {} result   {}'.format((i + 1), len(instances), (time.time() - start), result))
        if constraints is None:
            logging.info('<<<<< Done in {}, overall {} for instance #{}/#{}:{} is already of class {}'.
                         format((time.time() - start_instance), (time.time() - start), i, len(instances), instance,
                                required_class))
        else:
            logging.info('<<<<< Done in {}, overall {} for instance #{}/#{} {} closest point to {} in area {}'.
                         format((time.time() - start_instance), (time.time() - start), i, len(instances), instance,
                                result,
                                constraints.get_class_area()))
        if result is None:
            logging.info('<<<<<<!!!!! No closest point found for instance {} !!!!!!!!!!!!!'.format(instance))
            continue
        results.append(result[0])
    if pm.TRAIN_SET is not None:
        print('Starting testing prediction')
        X = np.array(results)
        y = np.array([required_class] * len(results))

        classifier = DeepHyperplanesClassifier(hp_states)
        hp_score = classifier.score(X, y)
        print('Name: [HYPL_CLASSIFIER], Score: [', hp_score, ']')

        ccc.load_and_test(pm.TRAIN_SET, X, y)
        print('Finished testing prediction')


def main():
    pm.load_params()
    execute()


if __name__ == "__main__":
    main()
