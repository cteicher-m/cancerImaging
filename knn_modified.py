# K-nearest neighbors modified with weights
# model is the entire training dataset

# laod dataset and randomly split set into training set and 
# test set

import math
import numpy as np
from scipy.spatial import distance 

# use Euclidean distance to measure differences
def euclideanDistance(loc1, loc2):
    total = 0
    diff = 0
    for i in range(len(loc1)):
        diff = loc2(i) - loc1(i)
        total += diff * diff
    return float(math.sqrt(total))

def findNeighborsWeighted(trainNumber, features, groundTruth, k):

    groundTruth = np.reshape(groundTruth,(322,))
    f = features
    gT = groundTruth
    
    # Trim lists for training
    features = features[:trainNumber,:]
    groundTruth = groundTruth[:trainNumber]
    
    # Trim lists for testing
    f = f[trainNumber:,:]
    gT = gT[trainNumber:]
    
    # return 2d matrix of each euclidean distance with respect to two indexes
    distances = distance.cdist(features, f, 'euclidean')
    sortedidxs = np.argsort(distances)
    # return k nearest indexes
    knn_idxs = sortedidxs[:k]
    # weighted vote based on number in k nearest and weighted by distance
    for val in knn_idxs:
        # weight distance 
        val = 1 / val 
    vote = np.sum(features[knn_idxs]) 
    if vote > len(features) / 2 :
        return 0.0
    else:
        return 1.0

