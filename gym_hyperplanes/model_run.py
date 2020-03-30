import gym_hyperplanes.engine.execution_engine as engine
import gym_hyperplanes.engine.params as pm
import logging

if __name__ == "__main__":
    logging.basicConfig(filename='model_run.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    pm.load_params()
    engine.execute()
