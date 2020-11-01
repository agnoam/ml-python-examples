from sklearn import metrics, neighbors
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import LabelEncoder
import joblib

class Labels(Enum):
    Unacceptable = 0
    Acceptable = 1
    Good = 2
    VeryGood = 3

# Converts the string features into numbers
def convert_features(X):
    X_copy = X.copy()

    le = LabelEncoder()
    for i in range(len(X_copy[0])):
        X_copy[:, i] = le.fit_transform(X_copy[:, i])

    return X_copy

# Converts the string labels into numbers
def convert_labels(y):
    y_copy = y.copy()

    label_mapping = {
        'unacc': Labels.Unacceptable.value,
        'acc': Labels.Acceptable.value,
        'good': Labels.Good.value,
        'vgood': Labels.VeryGood.value
    }

    y_copy = y_copy.replace(label_mapping)
    return y_copy

def create_knn_model(X_train, y_train):
    knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
    knn.fit(X_train, y_train) # Training the model
    return knn

def print_example(X, y, model):
    print('Example:')
    print('    Input data: ', X[20])
    print('    Actual value: ', y.values[20])
    print('    Predicted value: ', model.predict(X)[20])

def main():
    data = pd.read_csv('car.data')

    X = data[['buying', 'maint', 'safety']].values # Features
    y = data[['class']] # Labels

    print('converting dataset strings into numbers')
    X = convert_features(X)
    y = convert_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = create_knn_model(X_train, y_train) # Creates model by the training data

    print('testing model...')

    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)

    print('Predictions: ', predictions)
    print('Accuracy: ', accuracy)

    print_example(X, y, model)

    print('Saving Model...')
    filename = 'model.model'
    joblib.dump(model, filename)

if __name__ == '__main__':
    main()