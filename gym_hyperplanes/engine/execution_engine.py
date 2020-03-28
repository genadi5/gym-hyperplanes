import time

import gym_hyperplanes.engine.model_trainer as trainer
import gym_hyperplanes.engine.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.engine.execution_container import ExecutionContainer
from gym_hyperplanes.states.data_provider import TestDataProvider
from gym_hyperplanes.states.dataset_provider import DataSetProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig
from gym_hyperplanes.states.state_calc import StateManipulator


def create_config():
    return HyperplaneConfig(area_accuracy=pm.ACCURACY, hyperplanes=pm.HYPERPLANES,
                            pi_fraction=pm.PI_FRACTION, from_origin_delta_percents=pm.FROM_ORIGIN_DELTA_PERCENTS,
                            max_steps=pm.STEPS, max_steps_no_better_reward=pm.STEPS_NO_REWARD_IMPROVEMENTS)


def create_execution(iter, config, data=None, area=None, external_hp_state=None, external_hp_dist=None):
    if pm.DATA_FILE is not None:
        return ExecutionContainer(iter, config, DataSetProvider(pm.DATA_NAME, pm.DATA_FILE, config, data),
                                  external_hp_state, external_hp_dist, area)
    else:
        return ExecutionContainer(iter, config, TestDataProvider(config, data),
                                  external_hp_state, external_hp_dist, area)


def execute():
    iteration = 0
    executions = [create_execution(iteration, create_config())]
    starting_execution = executions[0]
    execution_name = starting_execution.get_data_provider().get_name()

    data_size_to_process = starting_execution.get_data_size()

    hp_states = []

    start = time.time()
    execution_start = time.time()
    while len(executions) > 0:
        execution = executions[0]
        if iteration != execution.get_deep_level():
            print('@@@@@@@@@@@@@ Iteration [{}] finished in [{}], overall time [{}]'.
                  format(iteration, round(time.time() - execution_start), round(time.time() - start)))
            execution_start = time.time()
        iteration = execution.get_deep_level()
        print('@@@@@@@@@@@@@ Running iteration [{}] during time [{}], executions [{}] with data [{}]'.
              format(iteration, round(time.time() - execution_start), len(executions), data_size_to_process))
        del executions[0]
        state_manipulator = StateManipulator(execution.get_data_provider(), execution.get_config())
        done = trainer.execute_hyperplane_search(state_manipulator, execution.get_config())

        data_size_to_process -= execution.get_data_size()
        missed_areas, hp_state = state_manipulator.get_hp_state(done)
        if hp_state is not None:
            hp_state.set_external_boundaries(execution.get_external_hp_state(),
                                             execution.get_external_hp_dist(), execution.get_external_area())
            hp_states.append(hp_state)
        if len(missed_areas.get_missed_areas()) > 0:
            new_iteration = execution.get_deep_level() + 1
            if new_iteration > pm.ITERATIONS:
                print('Reached max allowed iterations. Finished')
                break
            for missed_area, missed_area_data in missed_areas.get_missed_areas().items():
                new_execution = create_execution(new_iteration, create_config(), missed_area_data, missed_area,
                                                 missed_areas.get_hp_state(), missed_areas.get_hp_dist())
                executions.append(new_execution)
                data_size_to_process += new_execution.get_data_size()
        print('@@@@@@@@@@@@@ Finished an execution in iteration [{}], time [{}], still executions [{}] with data [{}]'.
              format(iteration, round(time.time() - start), len(executions), data_size_to_process))

    print('@@@@@@@@@@@@@ Finished execution in [{}]'.format(round(time.time() - start)))
    hs.save_hyperplanes_state(hp_states, pm.MODEL_FOLDER + '{}_result.txt'.format(execution_name))
