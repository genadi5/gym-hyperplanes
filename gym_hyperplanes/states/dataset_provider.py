import math
import os.path

import numpy as np
import pandas as pd
from sklearn import datasets

from gym_hyperplanes.states.data_provider import DataProvider


class DataSetProvider(DataProvider):
    def __init__(self, hp_config, data=None):
        super(DataSetProvider, self).__init__(hp_config)
        if data is None:
            data_files = self.get_file_name()
            if os.path.isfile(data_files):
                self.data = pd.read_csv(data_files, header=None)
            else:
                self.data = datasets.load_iris()
        else:
            self.data = data

        self.only_data = self.data.iloc[:, :-1]
        min_value = float(min(self.only_data))
        max_value = float(max(self.only_data))

        self.features_minimums = np.amin(self.only_data, axis=0)
        self.features_maximums = np.amax(self.only_data, axis=0)

        min_value = min_value - ((max_value - min_value) * hp_config.get_from_origin_delta_percents()) / 100
        if min_value < 0:
            min_value = 0
        max_value = max_value + ((max_value - min_value) * hp_config.get_from_origin_delta_percents()) / 100

        # thinking that all features are homogenious we assume that their values have approximately
        # same range
        self.max_distance_from_origin = math.sqrt(pow(max_value, 2) * self.only_data.shape[1])
        self.min_distance_from_origin = min_value  # calculate - actually we should be able to move from - to +

        print('for {} min value {} and max value {} '.
              format(self.get_name(), self.min_distance_from_origin, self.max_distance_from_origin))
        print('loaded {} instances from {}'.format(self.data.shape[0], self.get_file_name()))

    def get_name(self):
        return None

    def get_file_name(self):
        return None

    def get_features_size(self):
        return self.only_data.shape[1]

    def get_data_size(self):
        return self.data.shape[0]

    def get_only_data(self):
        return self.only_data

    def get_data(self):
        return self.data

    def get_label(self, ind):
        return self.data.iloc[ind, -1]

    def get_max_distance_from_origin(self):
        return self.max_distance_from_origin

    def get_min_distance_from_origin(self):
        return self.min_distance_from_origin

    def get_features_minimums(self):
        return self.features_minimums

    def get_features_maximums(self):
        return self.features_maximums
