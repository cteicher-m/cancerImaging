# Decision Trees
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def decisionTreeFunction(trainNumber, features, groundTruth):
    # Initialize classifier
    dTC = DecisionTreeClassifier(max_depth=None, random_state=None)
    
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
    dTC.fit(features, np.reshape(groundTruth,(trainNumber,)))
    
    # Passing in params to classifier
    DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, 
                           min_samples_split=2, min_samples_leaf=1, 
                           min_weight_fraction_leaf=0.0, max_features='auto', 
                           random_state=0, max_leaf_nodes=None, 
                           class_weight=None, presort=False)

    predictions = dTC.predict(f)
    predictionDict = {}
    for i in range(trainNumber,322):
        predictionDict[i] = predictions[i - trainNumber]
    return(predictionDict)
