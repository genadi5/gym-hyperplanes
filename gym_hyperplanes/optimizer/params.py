from argparse import ArgumentParser
from configparser import ConfigParser

MODEL_FILE = None
REQUIRED_CLASS = None
INSTANCE = None


def load_params():
    global MODEL_FILE
    global REQUIRED_CLASS
    global INSTANCE

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", help="Data file path")
    parser.add_argument("-r", "--required_class", dest="required_class", help="Required class")
    parser.add_argument("-c", "--configuration", dest="configuration", default=None, help="Configuration file")
    parser.add_argument("-i", "--instance", dest="instance", help="Instance features")
    args = parser.parse_args()
    MODEL_FILE = args.file
    REQUIRED_CLASS = args.required_class
    INSTANCE = None if args.instance is None else [float(i) for i in args.instance.split(',')]
    if args.configuration is not None:
        config = ConfigParser()
        config.read(args.configuration)
        if config.has_option('MODEL', 'model_file'):
            MODEL_FILE = config.get('MODEL', 'model_file')
        if config.has_option('MODEL', 'required_class'):
            REQUIRED_CLASS = config.get('MODEL', 'required_class')
        if config.has_option('MODEL', 'instance'):
            INSTANCE = [float(i) for i in config.get('MODEL', 'instance').split(',')]
