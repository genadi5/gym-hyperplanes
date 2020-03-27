import gym_hyperplanes.engine.model_trainer as trainer
import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.iris.iris_data_provider import IrisDataProvider
from gym_hyperplanes.pendigits.pen_data_provider import PenDataProvider
from gym_hyperplanes.states.data_provider import TestDataProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig
from gym_hyperplanes.states.state_calc import StateManipulator

iris = False
pen = True

START_HYPERPLANES = 20
MAX_STEPS = 2000
MAX_STEPS_NO_BETTER_REWARD = MAX_STEPS / 4


def create_config(iteration, start_hyperplanes=START_HYPERPLANES):
    return HyperplaneConfig(area_accuracy=95, hyperplanes=START_HYPERPLANES,
                            pi_fraction=12, from_origin_delta_percents=5,
                            max_steps=MAX_STEPS, max_steps_no_better_reward=MAX_STEPS_NO_BETTER_REWARD)


def create_provider(iter, config, data=None, area=None, external_hp_state=None, external_hp_dist=None):
    if iris:
        return iter, config, IrisDataProvider(config, data), external_hp_state, external_hp_dist, area
    elif pen:
        return iter, config, PenDataProvider(config, data), external_hp_state, external_hp_dist, area
    else:
        return iter, config, TestDataProvider(config, data), external_hp_state, external_hp_dist, area


def main():
    iteration = 0
    providers = [create_provider(iteration, create_config(iteration))]
    provider_name = providers[0][2].get_name()

    hp_states = []

    while len(providers) > 0:
        provider = providers[0]
        iteration = provider[0]
        print('Running iteration [{}] still providers {}'.format(iteration, len(providers)))
        del providers[0]
        state_manipulator = StateManipulator(provider[2], provider[1])
        done = trainer.execute_hyperplane_search(state_manipulator, provider[1])

        missed_areas, hp_state = state_manipulator.get_hp_state(done)
        if hp_state is not None:
            hp_state.set_external_boundaries(provider[3], provider[4], provider[5])
            hp_states.append(hp_state)
        if len(missed_areas.get_missed_areas()) > 0:
            new_iteration = provider[0] + 1
            if new_iteration > 10:
                print('Reached max allowed iterations. Finished')
                break
            for missed_area, missed_area_data in missed_areas.get_missed_areas().items():
                providers.append(create_provider(new_iteration, create_config(new_iteration), missed_area_data,
                                                 missed_area, missed_areas.get_hp_state(), missed_areas.get_hp_dist()))
        print('After executing provider in iteration [{}] still providers {}'.format(iteration, len(providers)))

    hs.save_hyperplanes_state(hp_states, '/downloads/hyperplanes/{}_result.txt'.format(provider_name))


if __name__ == "__main__":
    main()
