from argparse import ArgumentParser
from configparser import ConfigParser

MODEL_FILE = None
REQUIRED_CLASS = None
INSTANCES = None
TRAIN_SET = None
PENETRATION_DELTA = None


def get_instances(filename):
    with open(filename) as f:
        content = f.readlines()
    instances = [x.strip() for x in content]
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

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", help="Data file path")
    parser.add_argument("-r", "--required_class", dest="required_class", help="Required class")
    parser.add_argument("-c", "--configuration", dest="configuration", default=None, help="Configuration file")
    parser.add_argument("-i", "--instances", dest="instances", help="Instances file")
    parser.add_argument("-t", "--train_set", dest="train_set", help="Train set")
    parser.add_argument("-p", "--penetration_delta", dest="penetration_delta", help="Penetration Delta")
    args = parser.parse_args()
    MODEL_FILE = args.file
    REQUIRED_CLASS = args.required_class
    INSTANCES = None if args.instances is None else get_instances(args.instances)
    TRAIN_SET = args.train_set
    PENETRATION_DELTA = float(args.penetration_delta) if args.penetration_delta is not None else None
    if args.configuration is not None:
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
