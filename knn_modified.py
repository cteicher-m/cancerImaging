# K-nearest neighbors modified with weights
# model is the entire training dataset

# laod dataset and randomly split set into training set and 
# test set

import math
import numpy as np
from scipy.spatial import distance 


import numpy as np 
import csv
import matplotlib.pyplot as plt


def findNeighborsWeighted(trainNumber, features, groundTruth, k):

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

        totaldist_class0 = 0
        totaldist_class1 = 0

        for j in range(k):
            if Y_train[knn_idxs[j]] == 0.0:
                totaldist_class0 += distancestoXTrain[knn_idxs[j]]
            else:
                totaldist_class1 += distancestoXTrain[knn_idxs[j]]


        vote = np.sum(Y_train[knn_idxs])
        if np.sum(Y_train[knn_idxs]) == 0:
            results.append(0.0)
        elif np.sum(Y_train[knn_idxs]) == k:
            results.append(1.0)
        else:
            totaldist_class1 /= np.sum(Y_train[knn_idxs])
            totaldist_class0 /= (k-np.sum(Y_train[knn_idxs]))
            if totaldist_class1 <= totaldist_class0:
                results.append(1.0)
            else:
                results.append(0.0)

    predictionDict = {}
    for i in range(trainNumber,322):
        predictionDict[i] = results[i - trainNumber]
    return(predictionDict)



 
