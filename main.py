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

url = "http://archive.ics.uci.edu/ml/machine-learning-databases" \
      "/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Assign colum names to the dataset
names = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
         'compactness', 'concavity', 'concave points', 'symmetry', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


def ED(x1, x2, length):  # it is used for calculating euclidean distance
    """

    Args:
        x1:
        x2:
        length:

    Returns:

    """
    distance = 0
    for x in range(length):
        distance += np.square(x1[x] - x2[x])
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
stratify=y, test_size=0.8)

