import logging

import gym_hyperplanes.optimizer.optimizer_runner as optimizer
import gym_hyperplanes.optimizer.params as pm

if __name__ == "__main__":
    logging.basicConfig(filename='optimizer_run.log', format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.DEBUG)
    pm.load_params()
    print('Loaded model file {}'.format(pm.MODEL_FILE))
    print('Loaded required class {}'.format(pm.REQUIRED_CLASS))
    print('Loaded {} instances'.format(len(pm.INSTANCES)))
    optimizer.execute()
    print('Finished execution {} instance on class {} from model file {}'.
          format(len(pm.INSTANCES), pm.REQUIRED_CLASS, pm.MODEL_FILE))
