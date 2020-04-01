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
    no_rewards_improvement = math.ceil(steps / pm.STEPS_NO_REWARD_IMPROVEMENTS_PART)
    return HyperplaneConfig(area_accuracy=pm.ACCURACY, hyperplanes=pm.HYPERPLANES,
                            pi_fraction=pm.PI_FRACTION, from_origin_delta_percents=pm.FROM_ORIGIN_DELTA_PERCENTS,
                            max_steps=steps, max_steps_no_better_reward=no_rewards_improvement)


def create_execution(iter, config, data=None, boundaries=None):
    if pm.DATA_FILE is not None:
        return ExecutionContainer(iter, config, DataSetProvider(pm.DATA_NAME, pm.DATA_FILE, config, data), boundaries)
    else:
        return ExecutionContainer(iter, config, TestDataProvider(config, data), boundaries)


pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())


def execute_search(execution, config_file):
    logging.basicConfig(filename='model_run.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    pm.load_params(config_file)

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
    return hp_state, new_executions, data_size_to_process


def execute():
    iteration = 0
    executions = [create_execution(iteration, create_config(iteration))]
    starting_execution = executions[0]
    execution_name = starting_execution.get_data_provider().get_name()

    data_size_to_process = starting_execution.get_data_size()

    hp_states = []

    start = time.time()
    while len(executions) > 0:
        start_iteration = time.time()
        print('@@@@@@@@@@@@@ Starting iteration [{}] at time [{}] secs, executions [{}] with data [{}]'.
              format(iteration, round(time.time() - start), len(executions), data_size_to_process))

        data_size_to_process = 0
        futures = [pool_executor.submit(execute_search, execution, pm.CONFIG_FILE) for execution in executions]
        submited_executions = len(executions)
        executions = []
        finished_executions = 0
        for future in concurrent.futures.as_completed(futures):
            new_hp_state = future.result()[0]
            new_executions = future.result()[1]
            new_data_size_to_process = future.result()[2]
            if new_hp_state is not None:
                hp_states.append(new_hp_state)
            executions = executions + new_executions
            data_size_to_process += new_data_size_to_process
            finished_executions += 1
            print(
                '@@@@@@@@@@@@ Finished {}/{} executions in iteration {} in {} secs, {} executions on next iteration with data size {}'.
                format(finished_executions, submited_executions, iteration, round(time.time() - start_iteration),
                       len(executions), data_size_to_process))

        msg = '@@@@@@@@@@@@@ Finished an execution in iteration [{}] in [{}] secs, time [{}] secs, still executions [{}] with data [{}]'
        print(msg.format(iteration, round(time.time() - start_iteration), round(time.time() - start), len(executions),
                         data_size_to_process))
        iteration += 1

    print('@@@@@@@@@@@@@ Finished execution in [{}]'.format(round(time.time() - start)))
    hs.save_hyperplanes_state(hp_states, pm.MODEL_FOLDER + '{}_result.txt'.format(execution_name))
