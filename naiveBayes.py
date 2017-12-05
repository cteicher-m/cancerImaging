# Naive Bayes implementation
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import operator

def naiveBayesAlg(trainNumber, features, groundTruth):

    # Create distinct bins for each of the continous features
    def discretize(data):
        for k in range(len(data[0])):
            temp = [] 
            for i in range(len(data)):
                temp.append(data[i][k])
            bin_means = pd.qcut(temp, 25).codes
            for i in range(len(data)):
                data[i][k] = bin_means[i]
        return(data)

    # General probability for a certain classification or feature bin
    def occurrences(classification):
        total = len(classification)
        prob = dict(Counter(classification))
        for key in prob.keys():
            prob[key] = prob[key]/float(total)
        return prob

    # Main
    def naive_bayes(training, outcome, new_sample):
        classes = np.unique(outcome)
        rows, cols = np.shape(training)
        likelihoods = {}
        for cls in classes:
            likelihoods[cls] = defaultdict(list)

        class_probabilities = occurrences(outcome)

        for cls in classes:
            row_indices = np.where(outcome == cls)[0]
            subset = training[row_indices, :]
            r, c = np.shape(subset)
            for j in range(0,c):
                likelihoods[cls][j] += list(subset[:,j])

        for cls in classes:
            for j in range(0,cols):
                 likelihoods[cls][j] = occurrences(likelihoods[cls][j])

        results = {}
        for cls in classes:
            class_probability = class_probabilities[cls]
            for i in range(len(new_sample)):
                relative_values = likelihoods[cls][i]
                if new_sample[i] in relative_values.keys():
                    class_probability *= relative_values[new_sample[i]]
                else:
                    class_probability *= 0
                results[cls] = class_probability
                
        return(max(results.iteritems(), key=operator.itemgetter(1))[0])

    groundTruth = np.reshape(groundTruth,(322,))
    f = features
    gT = groundTruth

    # Trim lists for training
    features = features[:trainNumber,:]
    groundTruth = groundTruth[:trainNumber]

    # Trim lists for testing
    f = f[trainNumber:,:]
    gT = gT[trainNumber:]
    
    # Sort values into discrete bins
    features = discretize(features) 
    f = discretize(f)
    
    predictions = []
    for i in range(len(f)): 
        predictions.append(naive_bayes(features, groundTruth, f[i]))
    
    predictionDict = {}
    for i in range(trainNumber,322):
        predictionDict[i] = predictions[i - trainNumber]
    return(predictionDict)


    
    