import numpy as np
import pandas as pd
import random

data_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/Games/Games.txt'

data = pd.read_csv(data_file, header=None)
df = data.iloc[:, :-1]

a = np.array(df)
u = np.unique(a)

rows = []
for i in range(0, 20):
    row = []
    for j in range(0, 18):
        row.append(random.choice(u))
    rows.append(row)

data_set = np.asarray(rows)
np.savetxt("c:\\downloads\\foo.csv", data_set, fmt='%i', delimiter=",")
print('sss')
