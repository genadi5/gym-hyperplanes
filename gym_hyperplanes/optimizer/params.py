import logging
from argparse import ArgumentParser
from configparser import ConfigParser

MODEL_FILE = None
REQUIRED_CLASS = None
INSTANCES = None
TRAIN_SET = None
PENETRATION_DELTA = None
SOURCE_MODEL_FILE = None
CONSTRAINTS_FILE = None

FEATURE_BOUND_AREA = 'AREA'
FEATURE_BOUND_FEATURES = 'FEATURES'
FEATURE_BOUND_STRETCH = 'STRETCH'
FEATURE_BOUND = FEATURE_BOUND_AREA
FEATURE_BOUND_STRETCH_RATIO = 0.5


def get_instances(filename):
    with open(filename) as f:
        content = f.readlines()
    instances = [x.strip() for x in content if not x.startswith('#')]
    result = []
    for instance in instances:
        result.append([float(i) for i in instance.split(',')])
    return result


def load_params():
    global MODEL_FILE
    global REQUIRED_CLASS
    global INSTANCES
    global TRAIN_SET
    global PENETRATION_DELTA
    global FEATURE_BOUND
    global SOURCE_MODEL_FILE
    global CONSTRAINTS_FILE

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", help="Data file path")
    parser.add_argument("-r", "--required_class", dest="required_class", help="Required class")
    parser.add_argument("-c", "--configuration", dest="configuration", default=None, help="Configuration file")
    parser.add_argument("-i", "--instances", dest="instances", help="Instances file")
    parser.add_argument("-t", "--train_set", dest="train_set", help="Train set")
    parser.add_argument("-p", "--penetration_delta", dest="penetration_delta", help="Penetration Delta")
    parser.add_argument("-b", "--feature_bound", dest="feature_bound", default='AREA', help="Features bound")
    parser.add_argument("-m", "--source_model", dest="source_model", default=None, help="Source Model File")
    parser.add_argument("-g", "--constraints", dest="constraints", default=None,
                        help="File with features bounds and budget configuration")
    args = parser.parse_args()
    MODEL_FILE = args.file
    REQUIRED_CLASS = args.required_class
    INSTANCES = None if args.instances is None else get_instances(args.instances)
    TRAIN_SET = args.train_set
    PENETRATION_DELTA = float(args.penetration_delta) if args.penetration_delta is not None else None
    FEATURE_BOUND = args.feature_bound
    SOURCE_MODEL_FILE = args.source_model
    CONSTRAINTS_FILE = args.constraints
    if args.configuration is not None:
        logging.info('^^^^^ Loading from config {}'.format(args.configuration))
        config = ConfigParser()
        config.read(args.configuration)
        if config.has_option('MODEL', 'model_file'):
            MODEL_FILE = config.get('MODEL', 'model_file')
        if config.has_option('MODEL', 'required_class'):
            REQUIRED_CLASS = config.get('MODEL', 'required_class')
        if config.has_option('MODEL', 'instances'):
            INSTANCES = get_instances(config.get('MODEL', 'instances'))
        if config.has_option('MODEL', 'train_set'):
            TRAIN_SET = config.get('MODEL', 'train_set')
        if config.has_option('MODEL', 'penetration_delta'):
            PENETRATION_DELTA = float(config.get('MODEL', 'penetration_delta'))
        if config.has_option('MODEL', 'feature_bound'):
            FEATURE_BOUND = config.get('MODEL', 'feature_bound')
        if config.has_option('MODEL', 'source_model_file'):
            SOURCE_MODEL_FILE = config.get('MODEL', 'source_model_file')
    else:
        logging.info('^^^^^ No configuration file provided, using command line and default arguments')

    if (SOURCE_MODEL_FILE is None) and (FEATURE_BOUND == FEATURE_BOUND_STRETCH):
        logging.info('No source model file provided while set {} bound, switching to {} bound'.
                     format(FEATURE_BOUND_STRETCH, FEATURE_BOUND_AREA))

    logging.info('MODEL_FILE {}'.format(MODEL_FILE))
    logging.info('REQUIRED_CLASS {}'.format(REQUIRED_CLASS))
    logging.info('INSTANCES {}'.format(INSTANCES))
    logging.info('TRAIN_SET {}'.format(TRAIN_SET))
    logging.info('PENETRATION_DELTA {}'.format(PENETRATION_DELTA))
    logging.info('FEATURE_BOUND {}'.format(FEATURE_BOUND))
    logging.info('SOURCE_MODEL_FILE {}'.format(SOURCE_MODEL_FILE))
    logging.info('CONSTRAINTS_FILE {}'.format(CONSTRAINTS_FILE))
