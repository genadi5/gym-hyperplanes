import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.utils import shuffle


def run_classifiers(X_train, X_test, y_train, y_test, cls=''):
    names = ["Nearest Neighbors", "RBF SVM", "Random Forest", "Neural Net"]

    classifiers = [
        KNeighborsClassifier(weights='distance', n_jobs=-1),
        SVC(probability=True, class_weight='balanced'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced'),
        MLPClassifier(solver='lbfgs', hidden_layer_sizes=(64, 64, 64,), max_iter=10000)]

    X_train, y_train = shuffle(X_train, y_train)

    for name, clf in zip(names, classifiers):
        print('Starting evaluation of {} for class {}'.format(name, cls))
        clf.fit(X_train, y_train)
        # predict = clf.predict(X_test)
        # print('For class {} predicted {}'.format(cls, predict))
        score = clf.score(X_test, y_test)

        print('For class {} -  Name: [{}], Score: [{}]'.format(cls, name, score))


def load_split_and_test(data_file):
    print('Loading data from {}'.format(data_file))
    data = pd.read_csv(data_file, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)

    for cls in set(y_test):
        x_cls_test = X_test[y_test == cls]
        y_cls_test = y_test[y_test == cls]
        run_classifiers(X_train, x_cls_test, y_train, y_cls_test, cls)


def load_and_test(train_data_file, X_test, y_test):
    data = pd.read_csv(train_data_file, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    run_classifiers(X, X_test, y, y_test)


def load_train_and_test(train_data_file, test_data_file):
    test_data = pd.read_csv(test_data_file, header=None)

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    load_and_test(train_data_file, X_test, y_test)


def main():
    # train_data_file = '/UP/Teza/classoptimizer/pendigits/pendigits.tra'
    # test_data_file = '/UP/Teza/classoptimizer/pendigits/pendigits.tes'
    # load_train_and_test(train_data_file, test_data_file)
    # load_split_and_test('/UP/Teza/classoptimizer/iris/iris.data')

    # 0.6 - 0.7
    data_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/Games/Games.txt'
    # very poor
    # data_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/huge/huge.txt'
    # 0.6 - 0.7
    # data_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/demo/data.txt'
    # 1.0
    # data_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/iris/iris.data'
    # 0.97-0.99
    # data_file = '/UP/Teza/classoptimizer/gym-hyperplanes/gym_hyperplanes/data/pendigits/pendigits.tra'
    load_split_and_test(data_file)


if __name__ == "__main__":
    main()
