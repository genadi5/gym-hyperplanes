import logging
import operator

import numpy as np
import pandas as pd

import gym_hyperplanes.states.hyperplanes_state as hs


def make_area(array, powers):
    return np.bitwise_or.reduce(powers[array > 0])


def score(classifier, X, y):
    pred = classifier.predict(X)

    count = 0
    for i in range(min(len(pred), len(y))):
        if pred[i] == y[i]:
            count += 1

    return (count * 100) / max(len(pred), len(y))


class HyperplanesClassifier:
    def __init__(self, hyperplane_state):
        self.hyperplane_state = hyperplane_state
        self.powers = np.array([pow(2, i) for i in range(len(self.hyperplane_state.hp_dist))])

    def predict(self, X):
        boundaries = []
        for boundary in self.hyperplane_state.boundaries:  # for each boundary state build instance position
            b_calc = np.dot(X, boundary.get_hp_state())
            b_signs = b_calc - boundary.get_hp_dist()
            b_sides = (b_signs > 0).astype(int)
            b_powers = np.array([pow(2, i) for i in range(len(boundary.get_hp_dist()))])
            boundaries.append((boundary.get_class_area(), b_sides, b_powers))

        calc = np.dot(X, self.hyperplane_state.hp_state)
        signs = calc - self.hyperplane_state.hp_dist
        sides = (signs > 0).astype(int)

        result = []

        for i, side in enumerate(sides):
            if not self.is_in_bound(i, boundaries, result):
                continue
            key = make_area(side, self.powers)
            cls = None if key not in self.hyperplane_state.areas_to_classes \
                else self.hyperplane_state.areas_to_classes[key]
            # if got None meaning it is out of area
            result.append(None if cls is None else max(cls.items(), key=operator.itemgetter(1))[0])

        return np.array(result)

    def is_in_bound(self, index, boundaries, result):
        for boundary in boundaries:
            b_key = make_area(boundary[1][index], boundary[2])
            if b_key != boundary[0]:
                result.append(None)
                return False
        return True

    def score(self, X, y):
        return score(self, X, y)

    def is_supported_class(self, required_class):
        return self.hyperplane_state.is_supported_class(required_class)


class DeepHyperplanesClassifier:
    def __init__(self, hp_states):
        # we are looping in reverse to start from the most deep area since if point is inside deeper (narrow) area
        # it for sure inside shallower (broader) area
        sorted_hp_states = sorted(hp_states, key=lambda x: len(x.boundaries))
        self.classifiers = [HyperplanesClassifier(hp_state) for hp_state in sorted_hp_states]

    def predict(self, X, required_class=None):
        # each state id is isolated areas (on it own deep level) so there can't be that same point will
        # fall in several states
        # thus once first classified it to required class  we finished
        y = np.array([None] * X.shape[0])
        for classifier in self.classifiers:
            if required_class is not None and not classifier.is_supported_class(required_class):
                continue
            ind = np.where(y == None)  # indexes where y is not predicted yet
            if len(ind) == 0:
                break
            cX = X[y == None]  # corresponded instances
            cy = classifier.predict(cX)

            for index, i in enumerate(ind[0]):
                y[i] = cy[index]
        return np.array(y)

    def score(self, X, y):
        return score(self, X, y)


def test_pendigits():
    # result_file = '/UP/Teza/classoptimizer/model/iris_result.txt'
    result_file = '/UP/Teza/classoptimizer/model/pendigits_result.hs'
    # model_file = '/UP/Teza/classoptimizer/model/avila_result.hs'
    # model_file = '/UP/Teza/classoptimizer/model/shuttle_result.hs'

    hp_states = hs.load_hyperplanes_state(result_file)
    classifier = DeepHyperplanesClassifier(hp_states)

    # data_files = '/UP/Teza/classoptimizer/pendigits/pendigits.tra'
    data_files = '/UP/Teza/classoptimizer/pendigits/pendigits.tes'
    # data_files = '/UP/Teza/classoptimizer/avila/avila-tr.txt'
    # data_files = '/UP/Teza/classoptimizer/shuttle/shuttle.trn'
    # data_files = '/UP/Teza/classoptimizer/shuttle/shuttle.tst'
    # data_files = '/UP/Teza/classoptimizer/iris/iris.data'
    data = pd.read_csv(data_files, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    score = classifier.score(X, y)
    print(score)


def test():
    hp_state = hs.load_hyperplanes_state('/downloads/hyperplanes/result.txt')
    classifier = HyperplanesClassifier(hp_state)
    X = np.array([[20, 20], [40, 40], [80, 80]])
    y = np.array([1, 2, 2])
    pred = classifier.predict(X)
    score = classifier.score(X, y)
    print(pred)
    print(score)


def test_games():
    result_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/GamesSingle/Games_single.hs'

    hp_states = hs.load_hyperplanes_state(result_file)
    classifier = DeepHyperplanesClassifier(hp_states)

    data_files = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/GamesSingle/Games_single.txt'
    data = pd.read_csv(data_files, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print(classifier.score(X, y))


def main():
    logging.basicConfig(filename='classifier_run.log', format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.DEBUG)
    # logging.info('Classifier start')
    # test_pendigits()
    test_games()
    # logging.info('Classifier finished')


if __name__ == "__main__":
    main()
