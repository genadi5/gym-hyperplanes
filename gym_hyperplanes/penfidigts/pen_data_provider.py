import pandas as pd

from gym_hyperplanes.states.data_provider import DataProvider


class PenDataProvider(DataProvider):
    def __init__(self):
        super(IrisDataProvider, self).__init__()
        self.data = pd.read_csv('pendigits.tra')

    def get_features_size(self):
        return self.data.shape[1] - 1

    def get_data_size(self):
        return self.data.shape[0]

    def get_instance(self, ind):
        return self.data[ind]
