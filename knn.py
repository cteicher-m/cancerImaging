# K-nearest neighbors 
# model is the entire training dataset

# laod dataset and randomly split set into training set and 
# test set

import math
import numpy as np
from scipy.spatial import distance 


import numpy as np 
import csv
import matplotlib.pyplot as plt


def findNeighbors(trainNumber, features, groundTruth, k):

    results = []
    groundTruth = np.reshape(groundTruth,(322,))
    # f = features
    # gT = groundTruth

    # Trim lists for training
    X_train = features[:trainNumber]
    Y_train = groundTruth[:trainNumber]
    
    # Trim lists for testing
    X_test = features[trainNumber:]
    Y_test = groundTruth[trainNumber:]

    # return 2d matrix of each euclidean distance with respect to two indexes
  
    distances = distance.cdist(X_test, X_train, 'euclidean')
    # print("distances", distances)

    for i in range(len(X_test)):
        distancestoXTrain = distances[i]
        sortedidxs = np.argsort(distancestoXTrain)
        knn_idxs = sortedidxs[:k]

        vote = np.sum(Y_train[knn_idxs]) 
        if vote >= k/2.0:
            results.append(1.0)
        else:
            results.append(0.0)

    predictionDict = {}
    for i in range(trainNumber,322):
        predictionDict[i] = results[i - trainNumber]
    return(predictionDict)


