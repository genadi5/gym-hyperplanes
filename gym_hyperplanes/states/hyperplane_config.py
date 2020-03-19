class HyperplaneConfig:
    def __init__(self):
        self.hyperplanes = 5
        self.pi_fraction = 12  # pi / self.pi_fraction
        print('hyperplanes {} with rotation fraction {}'.format(self.hyperplanes, self.pi_fraction))

    def get_hyperplanes(self):
        return self.hyperplanes

    def get_rotation_fraction(self):
        return self.pi_fraction
