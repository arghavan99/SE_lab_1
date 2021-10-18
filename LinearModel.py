from sklearn import linear_model
from sklearn.metrics import accuracy_score


class LinearClassifiers():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def ridge_classifier(self, alpha=1):
        model = linear_model.RidgeClassifier(alpha=alpha)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        return accuracy_score(self.y_test, pred)

    def logistic_regression(self, penalty='l2'):
        model = linear_model.LogisticRegression(random_state=0, penalty=penalty)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        return accuracy_score(self.y_test, pred)