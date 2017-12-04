# K-nearest neighbors modified with weights
# model is the entire training dataset

# laod dataset and randomly split set into training set and 
# test set

import math
import numpy as np
from scipy.spatial import distance 

# use Euclidean distance to measure differences
def euclideanDistance(loc1, loc2):
    return distance.eucliean(loc1, loc2)

def findNeighborsWeighted(trainingset, testset, k):
    # return 2d matrix of each euclidean distance with respect to two indexes
    distances = cdist(trainingset, testset, 'euclidean')
    sortedidxs = np.argsort(distances)
    # return k nearest indexes
    knn_idxs = sortedidxs[:k]
    # weighted vote based on number in k nearest and weighted by distance
    for val in knn_idxs:
        # weight distance 
        val = 1 / val 
    vote = np.sum(trainingset[knn_idxs]) 
    if vote > 1/2 :
        return 1
    else:
        return 0

