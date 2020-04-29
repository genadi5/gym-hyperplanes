import math
import statistics

import numpy as np
import pandas as pd

from gym_hyperplanes.states.data_provider import DataProvider


class InstanceProbability:
    """
    Contains statistics on instances of the class in area where this class is dominant
    Can be used later on to calculate probability to choose the matching area
    For example there can be several areas of the same class where one area can be nearer to the
    origin instance but with lower probability of the class
    """

    def __init__(self, class_name, class_frequency, instances_amount):
        self.class_name = class_name
        self.class_frequency = class_frequency
        self.instances_amount = instances_amount

    def get_class_name(self):
        return self.class_name

    def get_class_frequency(self):
        return self.class_frequency

    def get_total_amount(self):
        return self.instances_amount

    def __str__(self):
        return "{}{}/{}".format('' if self.class_frequency == self.instances_amount else '#', self.class_frequency,
                                self.instances_amount)

    def __repr__(self):
        return "{}{}/{}".format('' if self.class_frequency == self.instances_amount else '#', self.class_frequency,
                                self.instances_amount)


class DataSetProvider(DataProvider):
    def __init__(self, data_name, data_file, hp_config, data=None):
        super(DataSetProvider, self).__init__(hp_config)
        self.data_name = data_name
        self.data_file = data_file
        if data is None:
            self.data = pd.read_csv(self.data_file, header=None)
        else:
            self.data = pd.DataFrame(data)

        provided_labels = list(self.data.iloc[:, -1])

        # in case we have several identical instances - we are grouping by instances and
        # we want to identify probability of class per instance
        # so each group is group if indexes where instance appears
        # this info will be provided in the output area calculations so that optimizer
        # will be able to select best area by his own policy
        self.groups = self.data.groupby(by=[i for i in range(0, self.data.shape[1] - 1)]).groups

        # collecting data rows without labels
        data_rows = []
        # collecting labels
        data_labels = []
        # collecting labels statistics
        # when we calculate area - count instances of classes we provide this statistic info
        # which can be used by optimizer to select best (according to his policy) area
        data_label_probability = []

        # key is instance, value is group of indexes where instance appears
        for key, value in self.groups.items():
            data_rows.append(list(key))

            # fetch classes which appear for the instance
            classes = list(np.take(provided_labels, value))
            try:
                # find most frequent class
                frequent_cls = statistics.mode(classes)
            except statistics.StatisticsError:
                # in case all classes are of the same frequency statistics.mode raises exception
                # in this case we take the first one - no matter which
                frequent_cls = classes[0]
            data_labels.append(frequent_cls)
            data_label_probability.append(InstanceProbability(frequent_cls, classes.count(frequent_cls), len(classes)))

        self.data_labels = np.array(data_labels)
        self.only_data = np.array(data_rows)
        self.data_label_probabilities = np.array(data_label_probability)
        min_value = float(np.amin(self.only_data))
        max_value = float(np.amax(self.only_data))

        # overall maximums and minimums pre features
        # we use this to bound search space
        self.features_minimums = np.amin(self.only_data, axis=0)
        self.features_maximums = np.amax(self.only_data, axis=0)

        min_value = min_value - ((max_value - min_value) * hp_config.get_from_origin_delta_percents()) / 100
        if min_value < 0:
            min_value = 0
        max_value = max_value + ((max_value - min_value) * hp_config.get_from_origin_delta_percents()) / 100

        # thinking that all features are homogenious we assume that their values have approximately
        # same range
        # we use these two below in order to bound hyperplanes move from/to origin
        self.max_distance_from_origin = math.sqrt(pow(max_value, 2) * self.only_data.shape[1])
        self.min_distance_from_origin = min_value  # calculate - actually we should be able to move from - to +

        print('for {} min value {} and max value {} '.
              format(self.get_name(), self.min_distance_from_origin, self.max_distance_from_origin))
        print('loaded {} instances from {}'.format(self.data.shape[0], self.data_file))

    def get_name(self):
        return self.data_name

    def get_features_size(self):
        return self.only_data.shape[1]

    def get_data_size(self):
        return self.data.shape[0]

    def get_actual_data_size(self):
        return self.only_data.shape[0]

    def get_only_data(self):
        return self.only_data

    def get_area_data(self, area_array):
        # extracting data by provided indexes
        # generally we use this when we want to extract which belongs to specific area
        # for future processing on next deep level
        instances = self.only_data[area_array]
        indexes = []
        for instance in instances:
            instance_indexes = self.groups.get(tuple(instance))
            indexes += list(instance_indexes.values)
        area_data = pd.DataFrame(np.array(self.data.iloc[indexes, :]))
        area_features_minimums = np.amin(instances, axis=0)
        area_features_maximums = np.amax(instances, axis=0)

        return area_data, area_features_minimums, area_features_maximums

    def get_labels(self):
        return self.data_labels

    def get_labels_probabilities(self):
        return self.data_label_probabilities

    def get_max_distance_from_origin(self):
        return self.max_distance_from_origin

    def get_min_distance_from_origin(self):
        return self.min_distance_from_origin

    def get_features_minimums(self):
        return self.features_minimums

    def get_features_maximums(self):
        return self.features_maximums
