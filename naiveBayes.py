# Naive Bayes implementation
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy as np
from collections import Counter, defaultdict

def naiveBayesAlg(trainNumber, features, groundTruth):

    def occurrences(list1):
        no_of_examples = len(list1)
        prob = dict(Counter(list1))
        print("prob")
        print(prob)
        for key in prob.keys():
            prob[key] = prob[key] / float(no_of_examples)
        return prob

    def naive_bayes(training, outcome, new_sample):
        classes = np.unique(outcome)
        rows, cols = np.shape(training)
        likelihoods = {}
        for cls in classes:
            likelihoods[cls] = defaultdict(list)

        class_probabilities = occurrences(outcome)

        for cls in classes:
            row_indices = np.where(outcome == cls)[0]
            subset      = training[row_indices, :]
            r, c        = np.shape(subset)
            for j in range(0,c):
                likelihoods[cls][j] += list(subset[:,j])

        for cls in classes:
            for j in range(0,cols):
                 likelihoods[cls][j] = occurrences(likelihoods[cls][j])


        results = {}
        for cls in classes:
            class_probability = class_probabilities[cls]
            for i in range(0,len(new_sample)):
                relative_values = likelihoods[cls][i]
                if new_sample[i] in relative_values.keys():
                    class_probability *= relative_values[new_sample[i]]
                else:
                    class_probability *= 0
                results[cls] = class_probability
        print results

    
    fullFeatures = features
    fullTruth = groundTruth

    groundTruth = np.reshape(groundTruth,(322,))
    f = features
    gT = groundTruth

    # Trim lists for training
    features = features[:trainNumber,:]
    groundTruth = groundTruth[:trainNumber]

    # Trim lists for testing
    f = f[trainNumber:,:]
    gT = gT[trainNumber:]
    
    naive_bayes(features, groundTruth, f)


    
    