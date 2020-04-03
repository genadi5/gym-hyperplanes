import concurrent.futures
import logging
import math
import os
import time

import gym_hyperplanes.engine.model_trainer as trainer
import gym_hyperplanes.engine.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.engine.execution_container import ExecutionContainer
from gym_hyperplanes.states.data_provider import TestDataProvider
from gym_hyperplanes.states.dataset_provider import DataSetProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig
from gym_hyperplanes.states.state_calc import StateManipulator


def create_config(iteration):
    steps = pm.STEPS if iteration > pm.ENTRY_LEVELS else pm.ENTRY_LEVEL_STEPS
    hyperplanes = pm.HYPERPLANES if iteration > pm.ENTRY_LEVELS else pm.ENTRY_LEVEL_HYPERPLANES
    no_rewards_improvement = math.ceil(steps / pm.STEPS_NO_REWARD_IMPROVEMENTS_PART)
    return HyperplaneConfig(area_accuracy=pm.ACCURACY, hyperplanes=hyperplanes,
                            pi_fraction=pm.PI_FRACTION, from_origin_delta_percents=pm.FROM_ORIGIN_DELTA_PERCENTS,
                            max_steps=steps, max_steps_no_better_reward=no_rewards_improvement)


def create_execution(iter, config, data=None, boundaries=None):
    if pm.DATA_FILE is not None:
        return ExecutionContainer(iter, config, DataSetProvider(pm.DATA_NAME, pm.DATA_FILE, config, data), boundaries)
    else:
        return ExecutionContainer(iter, config, TestDataProvider(config, data), boundaries)


WORKERS = os.cpu_count()
# WORKERS = WORKERS * 2

pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)


# pool_executor = concurrent.futures.ProcessPoolExecutor(6)


def execute_search(execution, config_file):
    logging.basicConfig(filename='model_run.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    pm.load_params(config_file)

    msg = '$$$$$$$$$$$$$ Starting {} iteration with {} hyperplanes on data size {} $$$$$$$$$$$$$'. \
        format(execution.get_deep_level(), execution.get_config().get_hyperplanes(),
               execution.data_provider.get_data_size())
    print(msg)
    logging.info(msg)

    state_manipulator = StateManipulator(execution.get_data_provider(), execution.get_config())
    done = trainer.execute_hyperplane_search(state_manipulator, execution.get_config())

    new_iteration = execution.get_deep_level() + 1
    complete = done or new_iteration > pm.ITERATIONS  # if this was last iteration we get complete state
    missed_areas, hp_state = state_manipulator.get_hp_state(complete)
    boundaries = execution.get_boundaries()
    if boundaries is None:
        boundaries = []
    if hp_state is not None:
        hp_state.set_boundaries(boundaries)
    new_executions = []
    data_size_to_process = 0
    if len(missed_areas.get_missed_areas()) > 0:
        if new_iteration <= pm.ITERATIONS:
            for missed_area, missed_area_data in missed_areas.get_missed_areas().items():
                boundary = hs.HyperplanesBoundary(missed_areas.get_hp_state(), missed_areas.get_hp_dist(), missed_area)
                new_execution = create_execution(new_iteration, create_config(new_iteration), missed_area_data,
                                                 [boundary] + boundaries)
                new_executions.append(new_execution)
                data_size_to_process += new_execution.get_data_size()
        else:
            print('Reached max allowed iterations. Finished')
    return hp_state, new_executions, data_size_to_process, execution.get_data_size()


def execute():
    start_iteration = 1
    starting_execution = create_execution(start_iteration, create_config(start_iteration))

    execution_name = starting_execution.get_data_provider().get_name()
    hp_states = []

    data_to_process = starting_execution.get_data_size()
    done_executions = 0
    start = time.time()
    execute_search(starting_execution, pm.CONFIG_FILE)
    executions = [pool_executor.submit(execute_search, starting_execution, pm.CONFIG_FILE)]
    while len(executions) > 0:
        print('@@@@@@@@@@@@@ At time [{}] secs done [{}], running [{}] with data size [{}]'.
              format(round(time.time() - start), done_executions, len(executions), data_to_process))

        submitted_executions = []

        removed_from_execution = 0
        added_new_executions = 0
        for execution in concurrent.futures.as_completed(executions):
            removed_from_execution += 1
            new_hp_state = execution.result()[0]
            new_executions = execution.result()[1]
            new_execution_data_to_process = execution.result()[2]

            data_to_process += new_execution_data_to_process
            data_to_process -= execution.result()[3]

            if new_hp_state is not None:
                hp_states.append(new_hp_state)
            if len(new_executions) > 0:
                submitted_executions += [pool_executor.submit(execute_search, new_execution, pm.CONFIG_FILE)
                                         for new_execution in new_executions]

                print('@@@@@@@@@@@@ At time {} secs adding {} with data size {} to remaining {} to total size {}'.
                      format(round(time.time() - start), len(new_executions), new_execution_data_to_process,
                             len(executions) - removed_from_execution + added_new_executions, data_to_process))
                added_new_executions += len(new_executions)  # current round add after print above
                print('@@@@@@@@@@@@ At time {} secs done {}, running {} with total size {}'.
                      format(round(time.time() - start), done_executions,
                             len(executions) - removed_from_execution + added_new_executions, data_to_process))
            done_executions += 1

        executions = submitted_executions

    print('@@@@@@@@@@@@@ Finished {} execution in [{}]'.format(done_executions, round(time.time() - start)))
    if pm.HYPERPLANES_FILE is not None:
        output_file = pm.HYPERPLANES_FILE
    else:
        output_file = '{}_result.txt'.format(execution_name)
    hs.save_hyperplanes_state(hp_states, pm.MODEL_FOLDER + output_file)
