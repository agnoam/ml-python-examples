''' Second lesson '''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Loads the iris 'features' (dataset)
iris = load_iris()

print('All data columns titles')
print(iris.feature_names) # All data columns titles
print('Flower types')
print(iris.target_names) # All flower types

print(iris.data[0]) # All data of the flower
print(iris.target[0]) # The type of the flower

test_ids = [0, 50, 65, 73, 129, 87, 91, 22]

# Remove the indexes from the dataset
training_target = np.delete(iris.target, test_ids)
training_data = np.delete(iris.data, test_ids, axis = 0)

# Generate test data from the exluded traning data
testing_targets = iris.target[test_ids]
testing_data = iris.data[test_ids]

# Classifier
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(training_data, training_target) # Training the classifier

return_arr = classifier.predict(testing_data)

for data in return_arr:
    print("returned Data is: ")
    print(iris.target_names[data])