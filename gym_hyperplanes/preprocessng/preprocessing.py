import random

import numpy as np


def games0000():
    instances = []
    instances += load0000('/UP/Teza/data/Games0000_1processed.txt')
    instances += load0000('/UP/Teza/data/Games0000_2processed.txt')
    instances += load0000('/UP/Teza/data/Games0000_3processed.txt')
    instances += load0000('/UP/Teza/data/Games0000_4processed.txt')

    random.shuffle(instances)
    with open('/UP/Teza/data/Games0000_all_data.txt', 'w') as f:
        f.writelines(instances)


def load0000(file_name):
    with open(file_name) as f:
        content = f.readlines()

    frequencies = [int(s.strip()) for s in content[0].strip().split(' ') if len(s.strip()) > 0]
    tables = content[1:]

    instances = []
    for index in range(0, len(tables), 3):
        row1 = [i.strip() for i in tables[index + 0].split(' ') if len(i.strip()) > 0]
        row2 = [i.strip() for i in tables[index + 1].split(' ') if len(i.strip()) > 0]
        row3 = [i.strip() for i in tables[index + 2].split(' ') if len(i.strip()) > 0]
        player1 = np.array([row1, row2, row3])
        player2 = player1.T
        rollout = [','.join(map(str, a)) for a in player1.tolist() + player2.tolist()]
        instance = ''
        delimiter = ''
        for r in rollout:
            instance += delimiter + r
            delimiter = ','
        for f in range(0, 3):
            frequency = frequencies[index + f]
            class_id = f + 1
            for i in range(0, frequency):
                instances.append(instance + "," + str(class_id) + '\n')
    return instances


def games0000_major():
    instances = []
    instances += load0000_major('/UP/Teza/data/Games0000_1processed.txt')
    instances += load0000_major('/UP/Teza/data/Games0000_2processed.txt')
    instances += load0000_major('/UP/Teza/data/Games0000_3processed.txt')
    instances += load0000_major('/UP/Teza/data/Games0000_4processed.txt')

    random.shuffle(instances)
    with open('/UP/Teza/data/Games0000_all_single.txt', 'w') as f:
        f.writelines(instances)


def load0000_major(file_name):
    with open(file_name) as f:
        content = f.readlines()

    frequencies = [int(s.strip()) for s in content[0].strip().split(' ') if len(s.strip()) > 0]
    tables = content[1:]

    instances = []
    for index in range(0, len(tables), 3):
        row1 = [i.strip() for i in tables[index + 0].split(' ') if len(i.strip()) > 0]
        row2 = [i.strip() for i in tables[index + 1].split(' ') if len(i.strip()) > 0]
        row3 = [i.strip() for i in tables[index + 2].split(' ') if len(i.strip()) > 0]
        player1 = np.array([row1, row2, row3])
        player2 = player1.T
        rollout = [','.join(map(str, a)) for a in player1.tolist() + player2.tolist()]
        instance = ''
        for r in rollout:
            instance += r + ','
        frequencies_of = frequencies[index:index + 3]
        ind_of_max = frequencies_of.index(max(frequencies_of))
        instance += 'A' if ind_of_max == 0 else ('B' if ind_of_max == 1 else 'C')
        instances.append(instance + '\n')
    return instances


def games0001():
    instances = []
    instances += load0001('/UP/Teza/data/Games0001_processed_all.txt')

    random.shuffle(instances)
    with open('/UP/Teza/data/Games0001_all_data.txt', 'w') as f:
        f.writelines(instances)


def load0001(file_name):
    with open(file_name) as f:
        tables = f.readlines()

    instances = []
    for index in range(0, len(tables), 3):
        rows = [[i.strip() for i in tables[index + 0].split(' ') if len(i.strip()) > 0],
                [i.strip() for i in tables[index + 1].split(' ') if len(i.strip()) > 0],
                [i.strip() for i in tables[index + 2].split(' ') if len(i.strip()) > 0]]
        frequencies = []
        player1rows = []
        player2rows = []
        for row in rows:
            frequencies.append(int(row[-1]))
            player1row = []
            player2row = []
            for part in range(0, len(row) - 1):
                parts = row[part].split(',')
                player1row.append(parts[0])
                player2row.append(parts[1])
            player1rows.append(player1row)
            player2rows.append(player2row)

        player1 = np.array(player1rows)
        player2 = np.array(player2rows)
        rollout = [','.join(map(str, a)) for a in player1.tolist() + player2.tolist()]
        instance = ''
        delimiter = ''
        for r in rollout:
            instance += delimiter + r
            delimiter = ','
        for f in range(0, 3):
            frequency = frequencies[f]
            class_id = f + 1
            for i in range(0, frequency):
                instances.append(instance + "," + str(class_id) + '\n')
    return instances


def games0001_major():
    instances = []
    instances += load0001_major('/UP/Teza/data/Games0001_processed_all.txt')

    random.shuffle(instances)
    with open('/UP/Teza/data/Games0001_all_single.txt', 'w') as f:
        f.writelines(instances)


def load0001_major(file_name):
    with open(file_name) as f:
        tables = f.readlines()

    instances = []
    for index in range(0, len(tables), 3):
        rows = [[i.strip() for i in tables[index + 0].split(' ') if len(i.strip()) > 0],
                [i.strip() for i in tables[index + 1].split(' ') if len(i.strip()) > 0],
                [i.strip() for i in tables[index + 2].split(' ') if len(i.strip()) > 0]]
        frequencies = []
        player1rows = []
        player2rows = []
        for row in rows:
            frequencies.append(int(row[-1]))
            player1row = []
            player2row = []
            for part in range(0, len(row) - 1):
                parts = row[part].split(',')
                player1row.append(parts[0])
                player2row.append(parts[1])
            player1rows.append(player1row)
            player2rows.append(player2row)

        player1 = np.array(player1rows)
        player2 = np.array(player2rows)
        rollout = [','.join(map(str, a)) for a in player1.tolist() + player2.tolist()]
        instance = ''
        for r in rollout:
            instance += r + ','
        ind_of_max = frequencies.index(max(frequencies))
        instance += 'A' if ind_of_max == 0 else ('B' if ind_of_max == 1 else 'C')
        instances.append(instance + '\n')
    return instances


def main():
    # games0000_major()
    games0001_major()


if __name__ == "__main__":
    main()
