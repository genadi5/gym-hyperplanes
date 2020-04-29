class HyperplaneConfig:
    def __init__(self, area_accuracy=95, hyperplanes=10, pi_fraction=6, from_origin_delta_percents=5,
                 max_steps=10000, max_steps_no_better_reward=10000):
        # area accuracy - there is a dominant class which takes at least accuracy% of
        # all instances inside the area
        self.area_accuracy = area_accuracy
        # number of hyperplanes used for separation
        self.hyperplanes = hyperplanes
        # the fraction of pi which we use to rotate hyperplane: pi / self.pi_fraction
        self.pi_fraction = pi_fraction
        # fraction of space boundary (bounded by max and min of features) to which translate hyperplane
        self.from_origin_delta_percents = from_origin_delta_percents

        # number of steps to execute in iteration
        self.max_steps = max_steps
        # number of steps during which no change happened
        self.max_steps_no_better_reward = max_steps_no_better_reward
        print('hyperplanes {} rotation fraction {}, area_accuracy {}, from_origin_delta_percents {}, max_steps {}'.
              format(self.hyperplanes, self.pi_fraction, self.area_accuracy, self.from_origin_delta_percents,
                     self.max_steps))

    def get_hyperplanes(self):
        return self.hyperplanes

    def get_rotation_fraction(self):
        return self.pi_fraction

    def get_area_accuracy(self):
        return self.area_accuracy

    def get_from_origin_delta_percents(self):
        return self.from_origin_delta_percents

    def get_max_steps(self):
        return self.max_steps

    def get_max_steps_no_better_reward(self):
        return self.max_steps_no_better_reward
