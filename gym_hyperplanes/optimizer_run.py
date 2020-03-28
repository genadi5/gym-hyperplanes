import gym_hyperplanes.optimizer.optimizer_runner as optimizer
import gym_hyperplanes.optimizer.params as pm

if __name__ == "__main__":
    pm.load_params()
    optimizer.execute()
