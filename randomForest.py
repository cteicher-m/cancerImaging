# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def randomForestFunction(trainNumber, features, groundTruth):
    # Initialize classifier
    rFC = RandomForestClassifier(max_depth=None, random_state=None)
    
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
    rFC.fit(features, np.reshape(groundTruth,(trainNumber,)))
    
    # Whatever this block does
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=None, min_impurity_split=None, min_samples_leaf=1,  min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=1, oob_score=False, random_state=0, verbose=0, warm_start=False)
    
    predictions = rFC.predict(f)
    predictionDict = {}
    for i in range(trainNumber,322):
        predictionDict[i] = predictions[i - trainNumber]
    return(predictionDict)
