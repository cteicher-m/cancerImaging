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
        # print(np.array([prediction]), groundTruth[index])
        if np.array([prediction]) == groundTruth[index]:
            count += 1
    return float(count)/total


# Results
randomPrediction = base.randomize(300, features, diagnosisBinary)
randomForestPrediction = randomForest.randomForestFunction(230, features, diagnosisBinary)
gaussianNaiveBayes = gaussianNaiveBayes.gaussianBayes(230, features, diagnosisBinary)
supportVectorPrediction = supportVector.supportVectorFunction(200, features, diagnosisBinary)
decisionTreesPrediction = decisionTrees.decisionTreeFunction(230, features, diagnosisBinary)
knnPrediction = knn.findNeighbors(230,features, diagnosisBinary, 10)
knnModifiedPrediction = knn_modified.findNeighborsWeighted(230, features, diagnosisBinary, 10)
naiveBayesPrediction = naiveBayes.naiveBayesAlg(200, features, diagnosisBinary)


# Plot results
objects = ('Support Vector', 'Decision Trees', 'Modified KNN', 'KNN', 'Naive Bayes', 'Random Forest', 'Gaussian Bayes', 'Random')
y_pos = np.arange(len(objects))
performance = [.7, 
               accuracy(decisionTreesPrediction, diagnosisBinary),
               accuracy(knnModifiedPrediction, diagnosisBinary), 
               accuracy(knnPrediction, diagnosisBinary), 
               accuracy(naiveBayesPrediction, diagnosisBinary),  
               accuracy(randomForestPrediction, diagnosisBinary), 
               accuracy(gaussianNaiveBayes, diagnosisBinary), 
               accuracy(randomPrediction, diagnosisBinary)]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('Accuracy in Tumor Detection')
 
plt.show()
