import numpy as np


def make_area(array, powers):
    return np.bitwise_or.reduce(powers[array > 0])


def find_indexes(unique_row, whole_data, labels, powers, areas):
    rows_indexes = np.where(whole_data == unique_row)[0]
    rows_indexes_counts = np.unique(rows_indexes, return_counts=True)
    indexes = rows_indexes_counts[0][rows_indexes_counts[1] == len(unique_row)]
    classes = np.take(labels, indexes)
    classes_counts = np.unique(classes, return_counts=True)
    classes = {cls: cnt for cls, cnt in zip(classes_counts[0], classes_counts[1])}

    area = make_area(unique_row, powers)
    areas[area] = classes


data_areas = dict()
data_labels = [1, 3, 2, 3, 3]
data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [9, 0, 3, 5], [5, 6, 7, 8]])
data_powers = np.array([pow(2, i) for i in range(len(data[0]))])

u = np.unique(data, axis=0)
np.apply_along_axis(find_indexes, 1, u, data, data_labels, data_powers, data_areas)
print(data_areas)
