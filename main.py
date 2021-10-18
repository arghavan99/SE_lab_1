from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Ensembles import *
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

    if cat == 2:
        clf = EnsembleAndDTClassifiers(x_train, y_train, x_test, y_test)
        print('choose among available ensemble models below:'
              '\n1. Random Forest'
              '\n2. Adaboost'
              '\n3. Decision Tree')
        num = int(input())
        if num == 1:
            print('input number of estimators:')
            n_estimators = int(input())
            test_acc = clf.random_forest(n_estimators=n_estimators)

        if num == 2:
            print('input number of estimators:')
            n_estimators = int(input())
            test_acc = clf.adaboost(n_estimators=n_estimators)

        if num == 3:
            test_acc = clf.decision_tree()