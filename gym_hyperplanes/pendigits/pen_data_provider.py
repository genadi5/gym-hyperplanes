import math
import os.path

import pandas as pd
from sklearn import datasets

from gym_hyperplanes.states.data_provider import DataProvider


class PenDataProvider(DataProvider):
    def __init__(self):
        super(PenDataProvider, self).__init__()
        pen_tra_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/pendigits/pendigits.tra'
        if os.path.isfile(pen_tra_file):
            self.data = pd.read_csv(pen_tra_file, header=None)
        else:
            self.data = datasets.load_digits()

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

        print('for pen min value {} and max value {} with delta {}'.format(self.min_distance_from_origin,
                                                                           self.max_distance_from_origin,
                                                                           self.distance_from_origin_delta_percents))
        print('loaded {} instances from iris.data'.format(self.data.shape[0]))

    def get_features_size(self):
        return self.data.shape[1] - 1

    def get_data_size(self):
        return self.data.shape[0]

    def get_instance(self, ind):
        return self.data.iloc[ind]

    def get_max_distance_from_origin(self):
        return self.max_distance_from_origin

    def get_min_distance_from_origin(self):
        return self.min_distance_from_origin

    def get_distance_from_origin_delta_percents(self):
        return self.distance_from_origin_delta_percents
