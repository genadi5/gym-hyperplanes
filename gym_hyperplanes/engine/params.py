import math
import time
from argparse import ArgumentParser
from configparser import ConfigParser

DATA_NAME = 'test_' + time.strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_').replace(':', '_').replace('-', '_')
DATA_FILE = None
ITERATIONS = 5
ACCURACY = 90
PI_FRACTION = 12
FROM_ORIGIN_DELTA_PERCENTS = 5
HYPERPLANES = 30
STEPS = 15000
STEPS_NO_REWARD_IMPROVEMENTS_PART = 3
STEPS_NO_REWARD_IMPROVEMENTS = math.ceil(STEPS / STEPS_NO_REWARD_IMPROVEMENTS_PART)


def load_params():
    global DATA_FILE
    global DATA_NAME
    global STEPS
    global STEPS_NO_REWARD_IMPROVEMENTS_PART
    global STEPS_NO_REWARD_IMPROVEMENTS
    global ITERATIONS
    global HYPERPLANES
    global ACCURACY
    global PI_FRACTION
    global FROM_ORIGIN_DELTA_PERCENTS

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="name",
                        default='test_' + time.strftime('%Y-%m-%d %H:%M:%S').
                        replace(' ', '_').replace(':', '_').replace('-', '_'), help="Test name")
    parser.add_argument("-f", "--file", dest="file", default=None, help="Data file path")
    parser.add_argument("-c", "--configuration", dest="configuration", default=None, help="Configuration file")
    parser.add_argument("-s", "--episode_steps", dest="episode_steps", default=10000, help="Number of steps in episode")
    parser.add_argument("-i", "--no_improvement_in_reward_part", default=4, dest="no_improvement_in_reward",
                        help="Number of steps wo improvement in reward as part of all steps")
    parser.add_argument("-d", "--deep_iterations", dest="deep_iterations", default=5,
                        help="Number of deep levels of space cuts")
    parser.add_argument("-p", "--hyperplanes", dest="hyperplanes", default=20, help="Number of hyperplanes")
    parser.add_argument("-a", "--accuracy", dest="accuracy", default=100, help="Accuracy in percents for each area")
    parser.add_argument("-r", "--rotation_fraction", dest="rotation_fraction", default=12,
                        help="Pi fraction for hyperplane rotation")
    parser.add_argument("-o", "--distance_delta_from_origin", dest="distance_delta_from_origin", default=5,
                        help="Distance delta in percents of hyperplane movement in values range")
    args = parser.parse_args()
    DATA_FILE = args.file
    DATA_NAME = args.name
    STEPS = int(args.episode_steps)
    STEPS_NO_REWARD_IMPROVEMENTS_PART = int(args.no_improvement_in_reward)
    STEPS_NO_REWARD_IMPROVEMENTS = math.ceil(STEPS / STEPS_NO_REWARD_IMPROVEMENTS_PART)
    ITERATIONS = int(args.deep_iterations)
    HYPERPLANES = int(args.hyperplanes)
    ACCURACY = int(args.accuracy)
    PI_FRACTION = int(args.rotation_fraction)
    FROM_ORIGIN_DELTA_PERCENTS = int(args.distance_delta_from_origin)
    if args.configuration is not None:
        config = ConfigParser()
        config.read(args.configuration)
        if config.has_option('DATA', 'data_file'):
            DATA_FILE = config.get('DATA', 'data_file')
        if config.has_option('DATA', 'data_name'):
            DATA_NAME = config.get('DATA', 'data_name')

        if config.has_option('EXECUTION', 'steps'):
            STEPS = int(config.get('EXECUTION', 'steps'))
        if config.has_option('EXECUTION', 'no_improvement_in_reward_part'):
            STEPS_NO_REWARD_IMPROVEMENTS_PART = int(config.get('EXECUTION', 'no_improvement_in_reward_part'))
        STEPS_NO_REWARD_IMPROVEMENTS = math.ceil(STEPS / STEPS_NO_REWARD_IMPROVEMENTS_PART)
        if config.has_option('EXECUTION', 'deep_iterations'):
            ITERATIONS = int(config.get('EXECUTION', 'deep_iterations'))

        if config.has_option('HYPERPLANES', 'hyperplanes'):
            HYPERPLANES = int(config.get('HYPERPLANES', 'hyperplanes'))
        if config.has_option('HYPERPLANES', 'accuracy'):
            ACCURACY = int(config.get('HYPERPLANES', 'accuracy'))
        if config.has_option('HYPERPLANES', 'rotation_fraction'):
            PI_FRACTION = int(config.get('HYPERPLANES', 'rotation_fraction'))
        if config.has_option('HYPERPLANES', 'distance_delta_from_origin'):
            FROM_ORIGIN_DELTA_PERCENTS = int(config.get('HYPERPLANES', 'distance_delta_from_origin'))
