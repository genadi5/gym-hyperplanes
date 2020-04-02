import gym_hyperplanes.optimizer.model_builder as mb
import gym_hyperplanes.optimizer.params as pm
import gym_hyperplanes.states.hyperplanes_state as hs


def execute():
    hp_states = hs.load_hyperplanes_state(pm.MODEL_FILE)
    required_class = hp_states[0].get_class_adapter()(pm.REQUIRED_CLASS)
    instance = pm.INSTANCE
    result = mb.find_closest_point(instance, required_class, hp_states)
    print(result)


def main():
    pm.load_params()
    execute()


if __name__ == "__main__":
    main()
