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

# import classifiers
import base # random case
import randomForest
import gaussianNaiveBayes
import knn
import knn_modified
import naiveBayes
import supportVector

# Load diagnosis data as numpy array
diagnosis = csv.reader(open('simpleData.txt'),delimiter = " ")
diagnosisBinary = []
for row in diagnosis:
    diagnosisBinary.append(row)
diagnosisBinary = np.array(diagnosisBinary).astype(np.float)


features = np.loadtxt('features.txt')
# print featres, features.shape --> 322 entries, 6 features


# use Euclidean distance to measure differences
def euclideanDistance(loc1, loc2, length):
    distance = 0
    for x in range(length):
        distance += pow((loc1[x] - loc2[x]), length)
    return math.sqrt(distance)

def getResult(trainNumber, features, groundTruth, k):

    results = []
    groundTruth = np.reshape(groundTruth,(322,))
    f = features
    gT = groundTruth

    # Trim lists for training
    features = features[:trainNumber,:]
    groundTruth = groundTruth[:trainNumber]
    
    # Trim lists for testing
    f = f[trainNumber:,:]
    gT = gT[trainNumber:]

    # print("features")
    # print(features)
    # print("f")
    # print(f)
    distances = []
    # return 2d matrix of each euclidean distance with respect to two indexes
    for i in range(len(f)):
        #print(i, euclideanDistance(features[i],f[i],6))
        distances.append(euclideanDistance(features[i],f[i],6))
    # distances = distance.cdist(features, f, 'euclidean')
    # print("distances", distances)
    sortedidxs = np.argsort(distances)
    # return k nearest indexes
    knn_idxs = sortedidxs[:k]
    print("indexes")
    print(knn_idxs)
    # vote based on number in k nearest
    # print(features)
    vote = np.sum(features[knn_idxs]) 
    if vote >= len(features) / 2:
        results.append(1.0)
    else:
        results.append(1.0)
    print(results)
    return results

def findNeighbors(trainNumber, features, groundTruth, k):
    for i in range(trainNumber):
        getResult(200, features[i], groundTruth[i], 2)


print(findNeighbors(200,features, diagnosisBinary, 10))
