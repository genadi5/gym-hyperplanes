import math

FULL_CIRCLE = 360


class StateCalculator:
    def __init__(self):
        self.xy = [[1, 1], [10, 10], [10, 60], [20, 70], [60, 60], [80, 80], [70, 10], [90, 10]]
        self.c = [1, 1, 2, 2, 1, 1, 2, 2]

        self.cos = dict()
        self.cossqr = dict()
        for i in range(0, 360, 45):
            cos = math.cos(i)
            self.cos[i] = cos
            self.cossqr[i] = cos * cos

    def calculate_reward(self, state):
        features = 2
        areas = dict()
        for i in range(0, int(len(state) / features)):  # 2 - two features
            h = str(i)
            for ind in range(0, len(self.c)):
                point = self.xy[ind]
                anglesValSum = 0  # all cosinuses should sum in one
                calc = 0
                for j in range(0, features - 1):
                    cos = self.cos[state[i * features + j]]
                    anglesValSum += cos
                    calc += point[j] * cos
                cos = 1 - anglesValSum
                calc += point[features - 1] * cos
                side = "-" + h if calc < state[i * features + features - 1] else "+" + h
                cls = set() if side not in areas else areas[side]
                cls.add(self.c[ind])

        count = 0
        for key, value in areas.items():
            if len(value) > 1:
                count -= 1
        return count

    def apply(self, state, action_index, action_direction, angle_delta, hyperplane_params_dimension):
        hyperplane_ind = int(action_index / hyperplane_params_dimension)
        hyperplane_start = hyperplane_ind * hyperplane_params_dimension
        result = 0
        for i in range(hyperplane_start, hyperplane_start + hyperplane_params_dimension - 1):
            result += self.cossqr[state[i]] if i != action_index \
                else self.cossqr[(state[i] + action_direction * angle_delta) % FULL_CIRCLE]
        return None if result > 1 else 0
