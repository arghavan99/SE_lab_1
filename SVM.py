from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
class Svm():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def svm_classifier(self, kernel ='rbf'):
        model = SVC(kernel=kernel)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        return accuracy_score(self.y_test, pred)