from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from LinearModel import *

def load_data():
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.33, random_state=42)


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load_data()
    print('Welcome to BreastCancerClassifier application! '
          '\nPlease choose your desired model category and enter a number:'
          '\n1. LinearModel'
          '\n2. EnsembleModels and DecisionTrees'
          '\n3. SVM')
    cat = int(input())

    if cat == 1:
        clf = LinearClassifiers(x_train, y_train, x_test, y_test)
        print('choose among available linear models below:'
              '\n1. Ridge Classifier'
              '\n2. Logistic Regression')
        num = int(input())
        if num == 1:
            print('input alpha:')
            alpha = float(input())
            test_acc = clf.ridge_classifier(alpha)
        elif num == 2:
            print('input penalty (l1 or l2 ?):')
            penalty = input()
            test_acc = clf.logistic_regression(penalty)

