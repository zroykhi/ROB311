#!/usr/bin/python
#  -*- coding: utf-8 -*-

"""
1. Load the Data.
2. Initialize the value of K.
3. For predicting the output class for the test data, iterate from 1st data
point to the total number of data points.
3.1 Calculate distance between test data and each row of training data by the
help of euclidean distance.
3.2 Sort the calculated distance in ascending order.
3.3 Get the top K rows from the sorted array.
3.4 Now find out the most frequent class of the rows.
3.5 Return the predicted class for the test data.
"""

__author__ = "Simonkey"
__credits__ = []
__version__ = "1.0.1"
__email__ = "simon.queyrut@ensta-paris.fr"
__status__ = "Development"
__name__ = "main"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import sklearn
import itertools
from sklearn.metrics import confusion_matrix

url = "http://archive.ics.uci.edu/ml/machine-learning-databases" \
      "/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Assign colum names to the dataset
names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
         'compactness', 'concavity', 'concave points', 'symmetry', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def euclidian_distance(X_test, X_train, length):  # it is used for calculating euclidean distance
    """

    Args:
        x1:
        x2:
        length:

    Returns:

    """
    distance = 0
    for x in range(length):
        x_test_x = X_test[x]
        x_train_x = X_train[x]

        if x_train_x == '?':
            x_train_x = 0
        if x_test_x == '?':
            x_test_x = 0
        distance += np.square(int(x_test_x) - int(x_train_x))
    return np.sqrt(distance)


# def knn(trainingSet, testInstance, k):  # here we are defining our model
#
#     distances = {}
#     sort = {}
#
#     length = testInstance.shape[1]
#
#     for x in range(len(trainingSet)):
#         dist = ED(testInstance, trainingSet.iloc[x], length)
#         distances[x] = dist[0]
#     sortdist = sorted(distances.items(), key=operator.itemgetter(1))
#     neighbors = []
#     for x in range(k):
#         neighbors.append(sortdist[x][0])
#     Votes = {}  # to get most frequent class of rows
#     for x in range(len(neighbors)):
#         response = trainingSet.iloc[neighbors[x]][-1]
#         if response in Votes:
#             Votes[response] += 1
#         else:
#             Votes[response] = 1
#     sortvotes = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True)
#     return (sortvotes[0][0], neighbors)


# testSet = [[6.8, 3.4, 4.8, 2.4]]
# test = pd.DataFrame(testSet)
# k = 1
# k1 = 3
# result, neigh = knn(dataset, test, k)
# result1, neigh1 = knn(dataset, test, k1)
# print(result)
# print(neigh)
# print(result1)
# print(neigh1)

## -----=-=-=-=-=-=-
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
stratify=y, test_size=0.2)


distance_list = []
test_sample_index = 0
# X_test = [X_test[0], X_test[1]]
class_list = []
k = 10

for test_sample in X_test:
    distance_list.append([])
    for train_data in X_train:
        distance_list[test_sample_index].append(euclidian_distance(test_sample, train_data, 9))
    y_train_copy = list(y_train.copy())  # making a copy to save original from deterioration
    sorted_distance_list, sorted_y_train_copy = \
        (list(t) for t in zip(*sorted(zip(distance_list[test_sample_index], y_train_copy))))

    # print(sorted_distance_list)
    # print("of len", len(sorted_distance_list))
    # print(sorted_y_train_copy)
    # print("of len", len(sorted_y_train_copy))
    # sorted_distance_list = distance_list[test_sample_index].sort()
    res = max(set(sorted_y_train_copy[:k]), key=sorted_y_train_copy[:k].count)
    # print("winner is", res)
    # print("for k = ", k)
    class_list.append(res)
    test_sample_index += 1

C = confusion_matrix(y_test, class_list)
class_names = ['Good', 'Bad']
plot_confusion_matrix(C, classes=class_names, normalize=True,title='Normalized Confusion Matrix')

# plt.figure()
# plt.plot_confusion_matrix(C, classes=class_names, normalize=True,title='Normalized Confusion Matrix')
#####
