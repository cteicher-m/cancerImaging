# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

def gaussianBayes(trainNumber, features, groundTruth):
    gNB = GaussianNB()
    
    groundTruth = np.reshape(groundTruth,(322,))
    f = features
    gT = groundTruth
    
    # Trim lists for training
    features = features[:trainNumber,:]
    groundTruth = groundTruth[:trainNumber]
    
    # Trim lists for testing
    f = f[trainNumber:,:]
    gT = gT[trainNumber:]
    
    # Fit on training data
    gNB.fit(features, np.reshape(groundTruth,(trainNumber,)))
    predictions = gNB.fit(features, np.reshape(groundTruth,(trainNumber,))).predict(f)
    predictionDict = {}
    
    for i in range(trainNumber,322):
        predictionDict[i] = predictions[i - trainNumber]
    return(predictionDict)