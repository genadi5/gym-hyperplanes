import numpy as np

import gym_hyperplanes.classifiers.classic_classification as ccc
import gym_hyperplanes.optimizer.model_builder as mb
import gym_hyperplanes.optimizer.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs


def execute():
    hp_states = hs.load_hyperplanes_state(pm.MODEL_FILE)
    required_class = hp_states[0].get_class_adapter()(pm.REQUIRED_CLASS)
    instances = pm.INSTANCES
    results = []
    for instance in instances:
        result = mb.find_closest_point(instance, required_class, hp_states)
        results.append(result)

    if pm.TRAIN_SET is not None:
        X = np.array(results)
        y = np.array([required_class] * len(results))
        ccc.load_and_test(pm.TRAIN_SET, X, y)


def main():
    pm.load_params()
    execute()


if __name__ == "__main__":
    main()
