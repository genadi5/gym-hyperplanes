import numpy as np


class DataProvider:
    def __init__(self):
        pass

    def get_features_size(self):
        pass

    def get_data_size(self):
        pass

    def get_instance(self, ind):
        pass

    def get_max_distance_from_origin(self):
        pass

    def get_min_distance_from_origin(self):
        pass

    def get_distance_from_origin_delta_percents(self):
        pass


class TestDataProvider(DataProvider):
    def __init__(self):
        super(TestDataProvider, self).__init__()
        self.data = np.array(
            [[1, 1, 1], [10, 10, 1], [10, 60, 2], [20, 70, 2], [60, 60, 1], [70, 70, 1], [70, 10, 2], [90, 10, 2]]
            # [[20, 20, 1], [30, 30, 1], [60, 50, 2], [70, 60, 2], [50, 10, 2], [60, 10, 2], [80, 10, 1], [90, 10, 1]]
            # [[10, 30, 1], [10, 50, 1], [40, 10, 2], [40, 30, 2], [40, 40, 2], [40, 90, 2], [80, 10, 1], [90, 10, 1]]
            # [[10, 30, 1], [10, 50, 1],
            #  [40, 10, 1], [70, 10, 1],
            #  [40, 30, 2], [70, 30, 2], [40, 50, 2], [70, 50, 2],
            #  [40, 90, 1], [70, 90, 1],
            #  [90, 10, 1], [90, 30, 1]]
        )

    def get_features_size(self):
        return self.data.shape[1] - 1

    def get_data_size(self):
        return self.data.shape[0]

    def get_instance(self, ind):
        return self.data[ind]

    def get_max_distance_from_origin(self):
        return 100

    def get_min_distance_from_origin(self):
        return 0

    def get_distance_from_origin_delta_percents(self):
        return 5
