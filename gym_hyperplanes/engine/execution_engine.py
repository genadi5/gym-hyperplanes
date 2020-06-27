import concurrent.futures
import logging
import math
import os
import sys
import time

import gym_hyperplanes.engine.keras_model_trainer as keras_trainer
import gym_hyperplanes.engine.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs
from gym_hyperplanes.engine.execution_container import ExecutionContainer
from gym_hyperplanes.states.dataset_provider import DataSetProvider
from gym_hyperplanes.states.hyperplane_config import HyperplaneConfig
from gym_hyperplanes.states.state_calc import StateManipulator

ENTRY_LEVEL_CONCURRENCY = 1


def create_config(iteration):
    steps = pm.STEPS if iteration > pm.ENTRY_LEVELS else pm.ENTRY_LEVEL_STEPS
    hyperplanes = pm.HYPERPLANES if iteration > pm.ENTRY_LEVELS else pm.ENTRY_LEVEL_HYPERPLANES
    no_rewards_improvement = math.ceil(steps / pm.STEPS_NO_REWARD_IMPROVEMENTS_PART)
    return HyperplaneConfig(area_accuracy=pm.ACCURACY, hyperplanes=hyperplanes,
                            pi_fraction=pm.PI_FRACTION, from_origin_delta_percents=pm.FROM_ORIGIN_DELTA_PERCENTS,
                            max_steps=steps, max_steps_no_better_reward=no_rewards_improvement)


def create_execution(iter, config, data=None, boundaries=None):
    return ExecutionContainer(iter, config, DataSetProvider(pm.DATA_NAME, pm.DATA_FILE, config, data), boundaries)


WORKERS = os.cpu_count()
WORKERS = math.ceil(WORKERS * 0.8)

pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)


def execute_search(execution, config_file):
    """
    Here a single search processed
    :param execution: - execution for search process
    :param config_file: configuration file
    :return:
    """
    logging.basicConfig(filename='model_run.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    pm.load_params(config_file)

    msg = '$$$$$$$$$$$$$ Starting iteration {} with {} hyperplanes on data size {} $$$$$$$$$$$$$'. \
        format(execution.get_deep_level(), execution.get_config().get_hyperplanes(),
               execution.data_provider.get_actual_data_size())
    print(msg)
    logging.info(msg)

    state_manipulator = StateManipulator(execution.get_data_provider(), execution.get_config())
    try:
        # Run Forrest, Run - execute search
        done = keras_trainer.execute_hyperplane_search(state_manipulator, execution.get_config())
    except Exception as e:
        print('%%%%%%%%%%%%%%%%%ERROR%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(e)
        print('%%%%%%%%%%%%%%%%%ERROR%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        done = False

    new_iteration = execution.get_deep_level() + 1
    complete = done or new_iteration > pm.ITERATIONS  # if this was last iteration we get complete state
    # extracting missed and finished areas
    missed_areas, hp_state = state_manipulator.get_hp_state(complete)
    boundaries = execution.get_boundaries()
    if boundaries is None:
        boundaries = []
    if hp_state is not None:
        # each hyperplane space/state actually only a single area from previous execution
        # and thus it has external boundaries from previous  execution
        # the very first hyperplane space/state does not have external boundaries
        hp_state.set_boundaries(boundaries)
    new_executions = []
    data_size_to_process = 0
    if len(missed_areas.get_missed_areas()) > 0:
        if new_iteration <= pm.ITERATIONS:
            # for every missed area we generate new execution which will have as external boundaries
            # the boundaries of this area
            for missed_area, missed_area_data in missed_areas.get_missed_areas().items():
                boundary = hs.HyperplanesBoundary(missed_areas.get_hp_state(), missed_areas.get_hp_dist(), missed_area)
                new_execution = create_execution(new_iteration, create_config(new_iteration), missed_area_data,
                                                 [boundary] + boundaries)
                new_executions.append(new_execution)
                data_size_to_process += new_execution.get_data_size()
        else:
            print('Reached max allowed iterations. Finished')
    return hp_state, new_executions, data_size_to_process, execution.get_data_size(), \
           state_manipulator.get_best_reward_ever(), done


def accept_execution(added_new_executions, data_to_process, done_executions, execution, executions, hp_states,
                     removed_from_execution, start, submitted_executions):
    """
    Absorbs data from 'execution' and submits new searches for areas which did not ended with
    enough accuracy separation. Each such area is actually new execution where boundaries of area
    will be external boundaries for each areas on next level
    :param added_new_executions: statistics, for prints
    :param data_to_process: statistics, for prints
    :param done_executions: statistics, for prints
    :param execution: current finished execution
    :param executions: all execution for process
    :param hp_states: hyperplanes states already finished where we add the new one from current execution
    :param removed_from_execution: statistics, for prints
    :param start:
    :param submitted_executions: statistics, for prints
    :return:
    """
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
    return data_to_process, done_executions, added_new_executions, removed_from_execution


def execute():
    start_overall = time.time()
    """
    First search executed in parallel by several processes and then best of result is 
    selected - this way we improve chances for better performance 
    """
    starting_executions = [create_execution(1, create_config(1)) for _ in range(0, ENTRY_LEVEL_CONCURRENCY)]
    data_to_process = starting_executions[0].get_data_size()
    execution_name = starting_executions[0].get_data_provider().get_name()
    # (hp_state, _, _, _, _, done) = execute_search(starting_executions[0], pm.CONFIG_FILE)

    """
    Actual execution of first round
    """
    executions = [pool_executor.submit(execute_search, starting_execution, pm.CONFIG_FILE)
                  for starting_execution in starting_executions]
    best_start_execution = None
    best_start_reward = None
    done = False
    """
    Selecting the best execution for first round
    """
    for execution in concurrent.futures.as_completed(executions):
        execution_reward = execution.result()[4]
        done = execution.result()[5]
        print('########### Got starting execution with reward {}'.format(execution_reward))
        if best_start_reward is None or execution_reward > best_start_reward:
            best_start_execution = execution
            best_start_reward = execution_reward

        if done:
            print('Execution done with reward {}'.format(best_start_reward))
            break

    print('########### Finally starting execution with reward {} in {} secs'.
          format(best_start_reward, round(time.time() - start_overall)))

    hp_states = []
    executions = []
    """
    Collecting result for first round
    """
    data_to_process, done_executions, added_new_executions, removed_from_execution = \
        accept_execution(0, data_to_process, 1, best_start_execution, executions,
                         hp_states, 0, start_overall, executions)
    if not done:
        start = time.time()
        while len(executions) > 0:
            print('@@@@@@@@@@@@@ At time [{}] secs done [{}], running [{}] with data size [{}]'.
                  format(round(time.time() - start), done_executions, len(executions), data_to_process))

            submitted_executions = []

            removed_from_execution = 0
            added_new_executions = 0
            for execution in concurrent.futures.as_completed(executions):
                data_to_process, done_executions, added_new_executions, removed_from_execution = \
                    accept_execution(added_new_executions, data_to_process, done_executions, execution, executions,
                                     hp_states, removed_from_execution, start, submitted_executions)

            executions = submitted_executions

        print(
            '@@@@@@@@@@@@@ Finished {} layered executions in [{}]'.format(done_executions, round(time.time() - start)))
        print('@@@@@@@@@@@@@ Finished {} execution in [{}]'.format(done_executions, round(time.time() - start_overall)))
    else:
        print('!!!!Nothing to execute since first execution is done!!!!')
    if pm.HYPERPLANES_FILE is not None:
        output_file = pm.HYPERPLANES_FILE
    else:
        output_file = '{}_result.txt'.format(execution_name)
    hs.save_hyperplanes_state(hp_states, output_file)
    sys.exit()
