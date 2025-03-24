import pandas as pd
from analyzer.metric import MetricCalculator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt


class ClassificationModel:

    def __init__(self):
        pass

    def LogisticRegression(self, X_train, X_test, y_train, y_test,
                           penalty='l1', solver='liblinear', max_iter=1000, C=0.1):
        model = MultiOutputClassifier(LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter, C=C))
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        print('__________')
        print("Коэффициенты регуляризации:")
        for i, estimator in enumerate(model.estimators_):
            print(f"Целевая переменная {y_test.columns[i]}:")
            print(*[f"{feature}: {coef:.2f}" for feature, coef in zip(X_train.columns, estimator.coef_.flatten())],
                  sep='\n')
            print('__________')
        return y_test, y_pred, model

    def GaussianNB(self, X_train, X_test, y_train, y_test):
        model = MultiOutputClassifier(GaussianNB())
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        return y_test, y_pred, model

    def KNeighborsClassifier(self, X_train, X_test, y_train, y_test,
                             n_neighbors=5, metric='euclidean'):
        model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric))
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        return y_test, y_pred, model

    def SVC(self, X_train, X_test, y_train, y_test,
            kernel='rbf'):
        model = MultiOutputClassifier(svm.SVC(kernel=kernel))
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        return y_test, y_pred, model

    def DecisionTreeClassifier(self, X_train, X_test, y_train, y_test,
                               max_depth=2, criterion='gini'):
        model = MultiOutputClassifier(tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion))
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
        for i, estimator in enumerate(model.estimators_):
            plt.figure(figsize=(12, 8))
            tree.plot_tree(estimator, feature_names=X_train.columns, class_names=[str(c) for c in estimator.classes_],
                           filled=True)
            plt.title(f"Дерево для целевой переменной {y_test.columns[i]}")
            plt.show()
        print(classification_report(y_test, y_pred))
        return y_test, y_pred, model