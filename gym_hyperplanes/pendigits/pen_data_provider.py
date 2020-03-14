from gym_hyperplanes.states.dataset_provider import DataSetProvider


class PenDataProvider(DataSetProvider):
    def __init__(self):
        super(PenDataProvider, self).__init__()

    def get_name(self):
        return 'PEN-DIGITS'

    def get_file_name(self):
        return '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/pendigits/pendigits.tra'
