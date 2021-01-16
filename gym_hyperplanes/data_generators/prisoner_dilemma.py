import numpy as np
import pandas as pd

max_range = 20
games_num = 1000
play_threshold = 0.5
all_games = []
for i in range(0, games_num):
    one = np.random.randint(0, max_range)
    two = np.random.randint(0, max_range)
    while two == one:
        two = np.random.randint(0, max_range)
    three = np.random.randint(0, max_range)
    while three == one or three == two:
        three = np.random.randint(0, max_range)
    four = np.random.randint(0, max_range)
    while four == one or four == two or four == three:
        four = np.random.randint(0, max_range)
    rewards = [one, two, three, four]
    rewards = sorted(rewards)
    # play = 'Cooperate' if (rewards[3] <= max_range * play_threshold) or (rewards[0] >= \
    #                       (max_range * (1 - play_threshold))) else 'Defect'
    # play = 'Cooperate' if 2 * rewards[2] <= rewards[3] - rewards[0] else 'Defect'
    play = 'Cooperate' if rewards[0] >= rewards[3] * play_threshold else 'Defect'
    game = [rewards[2], rewards[0], rewards[3], rewards[1], rewards[2], rewards[3], rewards[0], rewards[1], play]
    all_games.append(game)

print('*****************************************')
for i in range(0, games_num):
    print(all_games[i])
print('*****************************************')
df = pd.DataFrame(all_games)
df.to_csv('/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/teza_example/prisoner_dilemma.txt', index=False,
          header=False)
