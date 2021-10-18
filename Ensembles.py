from sklearn import ensemble
from sklearn.metrics import accuracy_score


class EnsembleAndDTClassifiers():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def random_forest(self, n_estimators = 100 ):
        model = ensemble.RandomForestClassifie(n_estimators = n_estimators)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        return accuracy_score(self.y_test, pred)

    def adaboost(self, n_estimators=50):
        model = ensemble.AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        return accuracy_score(self.y_test, pred)