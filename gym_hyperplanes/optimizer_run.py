import logging

import gym_hyperplanes.optimizer.optimizer_runner as optimizer
import gym_hyperplanes.optimizer.params as pm

if __name__ == "__main__":
    logging.basicConfig(filename='optimizer_run.log', format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.DEBUG)
    pm.load_params()
    optimizer.execute()
