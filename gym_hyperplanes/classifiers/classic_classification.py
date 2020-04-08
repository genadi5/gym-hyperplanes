import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def run_classifiers(X_train, X_test, y_train, y_test):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=30, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier()]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        print('Predicted {}'.format(predict))
        score = clf.score(X_test, y_test)

        print('Name: [', name, '], Score: [', score, ']')


def load_split_and_test(data_file):
    data = pd.read_csv(data_file, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    run_classifiers(X_train, X_test, y_train, y_test)


def load_and_test(data_file, X_test, y_test):
    data = pd.read_csv(data_file, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    run_classifiers(X, X_test, y, y_test)
