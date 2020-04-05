import numpy as np

import gym_hyperplanes.classifiers.classic_classification as ccc
import gym_hyperplanes.optimizer.model_builder as mb
import gym_hyperplanes.optimizer.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs


def calculate_destination(from_point, to_point, delta):
    # r - fraction of distance between from_point and to_point intermediate point
    # in our case r = 1 + delta
    # so equation
    # R = (1 - r) * from_point + r * to_point
    # will look as
    # R = (1 + delta) * to_point - delta * from_point
    return [(1 + delta) * t - delta * f for f, t in zip(from_point, to_point)]


def execute():
    hp_states = hs.load_hyperplanes_state(pm.MODEL_FILE)
    required_class = hp_states[0].get_class_adapter()(pm.REQUIRED_CLASS)
    instances = pm.INSTANCES
    penetration_delta = pm.PENETRATION_DELTA
    results = []

    for instance in instances:
        the_class, result = mb.find_closest_point(instance, required_class, hp_states)
        if not the_class:
            result = calculate_destination(instance, result, penetration_delta)
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
