import logging
import math
import time
from argparse import ArgumentParser
from configparser import ConfigParser

DATA_NAME = 'test_' + time.strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_').replace(':', '_').replace('-', '_')
DATA_FILE = None
CONFIG_FILE = None
HYPERPLANES_FILE = None
ITERATIONS = 10
ACCURACY = 95
PI_FRACTION = 6
FROM_ORIGIN_DELTA_PERCENTS = 10
HYPERPLANES = 5
# levels starting from 1
ENTRY_LEVELS = 1
ENTRY_LEVEL_HYPERPLANES = 20
ENTRY_LEVEL_STEPS = 1000
STEPS = 4000
MAX_EPISODE_STEPS = 4000
STEPS_NO_REWARD_IMPROVEMENTS_PART = 4
STEPS_NO_REWARD_IMPROVEMENTS = math.ceil(STEPS / STEPS_NO_REWARD_IMPROVEMENTS_PART)
ROTATION_TRANSLATION_MAX_RATIO = 10


def load_params(config_file_path=None):
    global DATA_FILE
    global CONFIG_FILE
    global DATA_NAME
    global HYPERPLANES_FILE

    global ENTRY_LEVELS
    global ENTRY_LEVEL_HYPERPLANES
    global ENTRY_LEVEL_STEPS
    global STEPS
    global STEPS_NO_REWARD_IMPROVEMENTS_PART
    global STEPS_NO_REWARD_IMPROVEMENTS
    global ITERATIONS
    global MAX_EPISODE_STEPS

    global HYPERPLANES
    global ACCURACY
    global PI_FRACTION
    global FROM_ORIGIN_DELTA_PERCENTS
    global ROTATION_TRANSLATION_MAX_RATIO

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="name",
                        default='test_' + time.strftime('%Y-%m-%d %H:%M:%S').
                        replace(' ', '_').replace(':', '_').replace('-', '_'), help="Test name")
    parser.add_argument("-f", "--file", dest="file", default=None, help="Data file path")
    parser.add_argument("-c", "--configuration", dest="configuration", default=None, help="Configuration file")
    parser.add_argument("-s", "--steps", dest="steps", default=4000, help="Number of Steps")
    parser.add_argument("-e", "--episode_steps", dest="episode_steps", default=4000, help="Number of steps in episode")
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
    parser.add_argument("-m", "--model_output_folder", dest="model_folder", help="Model output folder")
    parser.add_argument("-t", "--rotation_translation_max_ratio", default=10, dest="rotation_translation_max_ratio",
                        help="Rotation translation max ratio")
    args = parser.parse_args()
    DATA_FILE = args.file
    DATA_NAME = args.name

    STEPS = int(args.steps)
    STEPS_NO_REWARD_IMPROVEMENTS_PART = int(args.no_improvement_in_reward)
    STEPS_NO_REWARD_IMPROVEMENTS = math.ceil(STEPS / STEPS_NO_REWARD_IMPROVEMENTS_PART)
    MAX_EPISODE_STEPS = int(args.episode_steps)

    ITERATIONS = int(args.deep_iterations)
    HYPERPLANES = int(args.hyperplanes)
    ACCURACY = int(args.accuracy)
    PI_FRACTION = int(args.rotation_fraction)
    FROM_ORIGIN_DELTA_PERCENTS = int(args.distance_delta_from_origin)
    ROTATION_TRANSLATION_MAX_RATIO = int(args.rotation_translation_max_ratio)
    if config_file_path is not None or args.configuration is not None:
        CONFIG_FILE = config_file_path
        config = ConfigParser()
        config.read(config_file_path if config_file_path is not None else args.configuration)
        if config.has_option('DATA', 'data_file'):
            DATA_FILE = config.get('DATA', 'data_file')
        if config.has_option('DATA', 'data_name'):
            DATA_NAME = config.get('DATA', 'data_name')
        if config.has_option('DATA', 'hyperplanes_file'):
            HYPERPLANES_FILE = config.get('DATA', 'hyperplanes_file')

        if config.has_option('EXECUTION', 'entry_levels'):
            ENTRY_LEVELS = int(config.get('EXECUTION', 'entry_levels'))
        if config.has_option('EXECUTION', 'entry_level_hyperplanes'):
            ENTRY_LEVEL_HYPERPLANES = int(config.get('EXECUTION', 'entry_level_hyperplanes'))
        if config.has_option('EXECUTION', 'entry_level_steps'):
            ENTRY_LEVEL_STEPS = int(config.get('EXECUTION', 'entry_level_steps'))

        if config.has_option('EXECUTION', 'steps'):
            STEPS = int(config.get('EXECUTION', 'steps'))
        if config.has_option('EXECUTION', 'no_improvement_in_reward_part'):
            STEPS_NO_REWARD_IMPROVEMENTS_PART = int(config.get('EXECUTION', 'no_improvement_in_reward_part'))
        STEPS_NO_REWARD_IMPROVEMENTS = math.ceil(STEPS / STEPS_NO_REWARD_IMPROVEMENTS_PART)
        if config.has_option('EXECUTION', 'deep_iterations'):
            ITERATIONS = int(config.get('EXECUTION', 'deep_iterations'))
        if config.has_option('EXECUTION', 'rotation_translation_max_ratio'):
            ROTATION_TRANSLATION_MAX_RATIO = int(config.get('EXECUTION', 'rotation_translation_max_ratio'))
        if config.has_option('EXECUTION', 'episode_steps'):
            MAX_EPISODE_STEPS = int(config.get('EXECUTION', 'episode_steps'))

        if config.has_option('HYPERPLANES', 'hyperplanes'):
            HYPERPLANES = int(config.get('HYPERPLANES', 'hyperplanes'))
        if config.has_option('HYPERPLANES', 'accuracy'):
            ACCURACY = int(config.get('HYPERPLANES', 'accuracy'))
        if config.has_option('HYPERPLANES', 'rotation_fraction'):
            PI_FRACTION = int(config.get('HYPERPLANES', 'rotation_fraction'))
        if config.has_option('HYPERPLANES', 'distance_delta_from_origin'):
            FROM_ORIGIN_DELTA_PERCENTS = int(config.get('HYPERPLANES', 'distance_delta_from_origin'))

    logging.info('DATA_FILE {}'.format(DATA_FILE))
    logging.info('CONFIG_FILE {}'.format(CONFIG_FILE))
    logging.info('DATA_NAME {}'.format(DATA_NAME))
    logging.info('HYPERPLANES_FILE {}'.format(HYPERPLANES_FILE))
    logging.info('ENTRY_LEVELS {}'.format(ENTRY_LEVELS))
    logging.info('ENTRY_LEVEL_HYPERPLANES {}'.format(ENTRY_LEVEL_HYPERPLANES))
    logging.info('ENTRY_LEVEL_STEPS {}'.format(ENTRY_LEVEL_STEPS))
    logging.info('STEPS {}'.format(STEPS))
    logging.info('STEPS_NO_REWARD_IMPROVEMENTS_PART {}'.format(STEPS_NO_REWARD_IMPROVEMENTS_PART))
    logging.info('STEPS_NO_REWARD_IMPROVEMENTS {}'.format(STEPS_NO_REWARD_IMPROVEMENTS))
    logging.info('ITERATIONS {}'.format(ITERATIONS))
    logging.info('MAX_EPISODE_STEPS {}'.format(MAX_EPISODE_STEPS))

    logging.info('HYPERPLANES {}'.format(HYPERPLANES))
    logging.info('ACCURACY {}'.format(ACCURACY))
    logging.info('PI_FRACTION {}'.format(PI_FRACTION))
    logging.info('FROM_ORIGIN_DELTA_PERCENTS {}'.format(FROM_ORIGIN_DELTA_PERCENTS))
    logging.info('ROTATION_TRANSLATION_MAX_RATIO {}'.format(ROTATION_TRANSLATION_MAX_RATIO))
