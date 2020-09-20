from statistics import mean

import numpy as np
import pandas as pd

ACTIONS = 10
PLAYERS = 2
ACTION_RANGE = 100
INSTANCES = 1000
INSTANCES_SIZE = np.power(ACTIONS, PLAYERS) * PLAYERS
DATA_FILE = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/huge/huge.txt'

instances = pd.DataFrame(np.random.randint(0, ACTION_RANGE, size=(INSTANCES, INSTANCES_SIZE)))
classes = pd.DataFrame()
classes = pd.DataFrame(np.zeros(shape=(INSTANCES, 1)))
for i in range(0, INSTANCES):
    classes.iloc[i, 0] = int(instances.iloc[i, 0] / (ACTION_RANGE / ACTIONS))
# classes[0] = instances.apply(lambda row: int((sum(row)/len(row)) / (int(ACTION_RANGE/ACTIONS))))
# classes[0] = instances.apply(lambda row: int(mean(row) / (int(ACTION_RANGE/ACTIONS))))
# classes[0] = instances.apply(lambda row: int(sum(row) / len(row)) % ACTIONS)
# classes = pd.DataFrame(np.random.randint(0, ACTIONS, size=(INSTANCES, 1)))
classes[0] = classes[0].apply(lambda x: chr(65 + int(x)))
# df = instances.join(classes)
df = pd.concat([instances, classes], axis=1)
df.to_csv(DATA_FILE, index=False, header=False)
