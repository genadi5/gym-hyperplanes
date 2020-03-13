class HyperplaneConfig:
    def __init__(self):
        self.hyperplanes = 4
        self.pi_fraction = 4  # pi / self.pi_fraction

    def get_hyperplanes(self):
        return self.hyperplanes

    def get_rotation_fraction(self):
        return self.pi_fraction
