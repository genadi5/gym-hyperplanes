import operator

import numpy as np

import gym_hyperplanes.states.hyperplanes_state as hs


class HyperplanesClassifier:
    def __init__(self, hyperplane_state):
        self.hyperplane_state = hyperplane_state

    def predict(self, X):
        calc = np.dot(X, self.hyperplane_state.hp_state)
        signs = calc - self.hyperplane_state.hp_dist
        sides = (signs > 0).astype(int)

        result = []

        for i, side in enumerate(sides):
            key = 0
            for k, val in enumerate(side):
                if val:
                    key |= 1 << k
            cls = 'UNKNOWN' if key not in self.hyperplane_state.areas_to_classes \
                else self.hyperplane_state.areas_to_classes[key]
            result.append(max(cls.items(), key=operator.itemgetter(1))[0])

        return np.array(result)

    def score(self, X, y):
        pred = self.predict(X)

        count = 0
        for i in range(min(len(pred), len(y))):
            if pred[i] == y[i]:
                count += 1

        return (count * 100) / max(len(pred), len(y))


def main():
    hp_state = hs.load_hyperplanes_state('/downloads/hyperplanes/result.txt')
    classifier = HyperplanesClassifier(hp_state)
    X = np.array([[20, 20], [40, 40], [80, 80]])
    y = np.array([1, 2, 2])
    pred = classifier.predict(X)
    score = classifier.score(X, y)
    print(pred)
    print(score)


if __name__ == "__main__":
    main()
