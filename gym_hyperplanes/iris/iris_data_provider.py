from gym_hyperplanes.states.dataset_provider import DataSetProvider


class IrisDataProvider(DataSetProvider):
    def __init__(self):
        super(IrisDataProvider, self).__init__()

    def get_name(self):
        return 'IRIS'

    def get_file_name(self):
        return '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/iris/iris.data'
