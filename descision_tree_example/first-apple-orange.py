from sklearn import tree

''' First lesson (apple vs orange) '''
# Constants variables
APPLE = 0
ORANGE = 1

# Classifiers data to our machine-learning program (input)
features = [
    [140, APPLE], # [grams, texture]
    [130, APPLE],
    [150, ORANGE],
    [170, ORANGE]
]

# What to print for each classifier match (output)
labels = [APPLE, APPLE, ORANGE, ORANGE]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels) # fit function = find patterns in our classifier data

if classifier.predict([[190, APPLE]]) == APPLE:
    print('This is apple')
else:
    print('This is orange')