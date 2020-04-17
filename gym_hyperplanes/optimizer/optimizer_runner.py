import time

import gym_hyperplanes.classifiers.classic_classification as ccc
import gym_hyperplanes.optimizer.model_builder as mb
import gym_hyperplanes.optimizer.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs
import numpy as np
from gym_hyperplanes.classifiers.hyperplanes_classifier import DeepHyperplanesClassifier

np.set_printoptions(precision=pm.PRECISION)


def execute():
    hp_states = hs.load_hyperplanes_state(pm.MODEL_FILE)
    required_class = hp_states[0].get_class_adapter()(pm.REQUIRED_CLASS)
    instances = pm.INSTANCES
    penetration_delta = pm.PENETRATION_DELTA
    results = []

    start = time.time()
    for i, instance in enumerate(instances):
        print('>>>>> #{}/#{} at time {} instance {}'.format(i, len(instances), (time.time() - start), instance))
        start_instance = time.time()
        result, constraints = mb.find_closest_point(instance, required_class, hp_states, penetration_delta)
        print('<<<<< Done in {}, overall {} for instance #{}/#{} {} closest point {} in constraint {}'.
              format((time.time() - start_instance), (time.time() - start), i, len(instances), instance, result,
                     constraints))
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
