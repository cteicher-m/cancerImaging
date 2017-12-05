# Support Vector Machine Classification
from sklearn import svm
import numpy as np

def supportVectorFunction(trainNumber, features, groundTruth):
    # Initialize classifier
    svmC = svm.SVC()
    
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
    svmC.fit(features, np.reshape(groundTruth,(trainNumber,)))
    
    # Passing in params to classifier
    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', 
        kernel='rbf', max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001, verbose=False)

    predictions = svmC.predict(f)
    predictionDict = {}
    for i in range(trainNumber,322):
        predictionDict[i] = predictions[i - trainNumber]
    return(predictionDict)