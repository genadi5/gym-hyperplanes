import math
import os.path

import pandas as pd
from sklearn import datasets

from gym_hyperplanes.states.data_provider import DataProvider


class IrisDataProvider(DataProvider):
    def __init__(self):
        super(IrisDataProvider, self).__init__()
        iris_data_files = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/iris/iris.data'
        if os.path.isfile(iris_data_files):
            self.data = pd.read_csv(iris_data_files)
        else:
            self.data = datasets.load_iris()

        only_data = self.data.iloc[:, 0:-1]
        min_value = float(min(only_data))
        max_value = float(max(only_data))

        self.distance_from_origin_delta_percents = 5

        min_value = min_value - ((max_value - min_value) * self.distance_from_origin_delta_percents) / 100
        if min_value < 0:
            min_value = 0
        max_value = max_value + ((max_value - min_value) * self.distance_from_origin_delta_percents) / 100

        self.max_distance_from_origin = math.sqrt(pow(max_value, 2) * only_data.shape[1])
        self.min_distance_from_origin = min_value  # calculate - actually we should be able to move from - to +

        print('loaded {} instances from iris.data'.format(self.data.shape[0]))

    def get_features_size(self):
        return self.data.shape[1] - 1

    def get_data_size(self):
        return self.data.shape[0]

    def get_instance(self, ind):
        return self.data[ind]

    def get_max_distance_from_origin(self):
        return self.max_distance_from_origin

    def get_min_distance_from_origin(self):
        return self.min_distance_from_origin

    def get_distance_from_origin_delta_percents(self):
        return self.distance_from_origin_delta_percents
