from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

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

