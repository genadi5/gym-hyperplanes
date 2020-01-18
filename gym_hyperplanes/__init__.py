from gym.envs.registration import register

register(
    id='hyperplanes-v0',
    entry_point='gym_hyperplanes.envs:HyperPlanesEnv',
)
