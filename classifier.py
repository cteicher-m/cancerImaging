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
# import supportVector
# import decisionTrees


# Load diagnosis data as numpy array
diagnosis = csv.reader(open('simpleData.txt'),delimiter = " ")
diagnosisBinary = []
for row in diagnosis:
    diagnosisBinary.append(row)
diagnosisBinary = np.array(diagnosisBinary).astype(np.float)
# print(sum(diagnosisBinary)) --> 115 positive for tumor
# print diagnosisBinary, diagnosisBinary.shape --> 322 entries, values 0 or 1

# Load feature data as numpy array
features = np.loadtxt('features.txt')
# print featres, features.shape --> 322 entries, 6 features

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
randomForestPrediction = randomForest.randomForestFunction(200, features, diagnosisBinary)
gaussianNaiveBayes = gaussianNaiveBayes.gaussianBayes(200, features, diagnosisBinary)
# supportVectorPrediction = supportVector.supportVectorFunction(200, features, diagnosisBinary)
# decisionTreesPrediction = decisionTrees.decisionTreeFunction(200, features, diagnosisBinary)
knnPrediction = knn.findNeighbors(200,features, diagnosisBinary, 10)
knnModifiedPrediction = knn_modified.findNeighborsWeighted(200,features, diagnosisBinary, 10)
naiveBayesPrediction = naiveBayes.naiveBayesAlg(200, features, diagnosisBinary)




# Plot results
objects = ('Support Vector', 'Modified KNN', 'KNN', 'Naive Bayes', 'Random Forest', 'Gaussian Bayes', 'Random')
# performance = [accuracy(supportVectorPrediction, diagnosisBinary), accuracy(knnModifiedPrediction, diagnosisBinary), accuracy(knnPrediction, diagnosisBinary), accuracy(naiveBayesPrediction, diagnosisBinary),  accuracy(randomForestPrediction, diagnosisBinary), accuracy(gaussianNaiveBayes, diagnosisBinary), accuracy(randomPrediction, diagnosisBinary)]
 
# objects = ('Random Forest', 'Naive Bayes', 'Random')
y_pos = np.arange(len(objects))
performance = [accuracy(randomForestPrediction, diagnosisBinary),
               accuracy(gaussianNaiveBayes, diagnosisBinary),
               accuracy(randomPrediction, diagnosisBinary),
               accuracy(knnModifiedPrediction, diagnosisBinary),
               accuracy(knnPrediction, diagnosisBinary)]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('Accuracy in Tumor Detection')
 
plt.show()
