from gym_hyperplanes.states.dataset_provider import DataSetProvider


class PenDataProvider(DataSetProvider):
    def __init__(self, hp_config, data=None):
        super(PenDataProvider, self).__init__(hp_config, data)

    def get_name(self):
        return 'PEN_DIGITS'

    def get_file_name(self):
        return '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/pendigits/pendigits.tra'
