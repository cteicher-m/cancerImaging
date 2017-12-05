# Tuning parameters

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
import decisionTrees


# Load diagnosis data as numpy array
diagnosis = csv.reader(open('simpleData.txt'),delimiter = " ")
diagnosisBinary = []
for row in diagnosis:
    diagnosisBinary.append(row)
diagnosisBinary = np.array(diagnosisBinary).astype(np.float)

# Load feature data as numpy array
features = np.loadtxt('features.txt')

def accuracy(predictions, groundTruth):
    total = len(predictions)
    count = 0
    for index, prediction in predictions.iteritems():
        if np.array([prediction]) == groundTruth[index]:
            count += 1
    return float(count)/total


# Choosing training number
# Yields around 230
numberVector = []
accuracyVectorRF = []
accuracyVectorGNB = []
accuracyVectorDT = []

for i in range(1,7):
    numberVector.append(i*46)
    aRF = accuracy(randomForest.randomForestFunction(i*46, features, diagnosisBinary), diagnosisBinary)
    accuracyVectorRF.append(aRF)
    aGNB = accuracy(gaussianNaiveBayes.gaussianBayes(i*46, features, diagnosisBinary), diagnosisBinary)
    accuracyVectorGNB.append(aGNB)
    aDT = accuracy(decisionTrees.decisionTreeFunction(i*46, features, diagnosisBinary), diagnosisBinary)
    accuracyVectorDT.append(aDT)

    
plt.plot(numberVector, accuracyVectorRF, 'b-', label='Random Forest')
plt.plot(numberVector, accuracyVectorGNB, 'g-', label='Gaussian Naive Bayes')
plt.plot(numberVector, accuracyVectorDT, 'r-', label='Decision Trees')
plt.ylabel("Accuracy")
plt.xlabel("Training Number")
plt.show()

# Number of estimators for random forest classifier
# Yields around 22
numberVector = []
accuracyVectorEstimator = []

for i in range(1,25):
    numberVector.append(i*2)
    aRFE = accuracy(randomForest.tuningFunction(i*2, 230, features, diagnosisBinary), diagnosisBinary)
    accuracyVectorEstimator.append(aRFE)

plt.plot(numberVector, accuracyVectorEstimator, label='Random Forest')
plt.ylabel("Accuracy")
plt.xlabel("Number of Estimators")
plt.show()
