import math
import os.path

import pandas as pd
from sklearn import datasets

from gym_hyperplanes.states.data_provider import DataProvider


class DataSetProvider(DataProvider):
    def __init__(self):
        super(DataSetProvider, self).__init__()
        data_files = self.get_file_name()
        if os.path.isfile(data_files):
            self.data = pd.read_csv(data_files, header=None)
        else:
            self.data = datasets.load_iris()

        self.only_data = self.data.iloc[:, :-1]
        min_value = float(min(self.only_data))
        max_value = float(max(self.only_data))

        self.distance_from_origin_delta_percents = 5

        min_value = min_value - ((max_value - min_value) * self.distance_from_origin_delta_percents) / 100
        if min_value < 0:
            min_value = 0
        max_value = max_value + ((max_value - min_value) * self.distance_from_origin_delta_percents) / 100

        self.max_distance_from_origin = math.sqrt(pow(max_value, 2) * self.only_data.shape[1])
        self.min_distance_from_origin = min_value  # calculate - actually we should be able to move from - to +

        print(
            'for {} min value {} and max value {} with delta {}'.format(self.get_name(), self.min_distance_from_origin,
                                                                        self.max_distance_from_origin,
                                                                        self.distance_from_origin_delta_percents))
        print('loaded {} instances from {}'.format(self.data.shape[0], self.get_file_name()))

    def get_name(self):
        return None

    def get_file_name(self):
        return None

    def get_features_size(self):
        return self.only_data.shape[1]

    def get_data_size(self):
        return self.data.shape[0]

    def get_data(self):
        return self.only_data

    def get_label(self, ind):
        return self.data.iloc[ind, -1]

    def get_max_distance_from_origin(self):
        return self.max_distance_from_origin

    def get_min_distance_from_origin(self):
        return self.min_distance_from_origin

    def get_distance_from_origin_delta_percents(self):
        return self.distance_from_origin_delta_percents
