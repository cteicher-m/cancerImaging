import numpy as np 
import csv
import matplotlib.pyplot as plt

# import classifiers
import base # random case
import randomForest
import gaussianNaiveBayes
import knn
import naiveBayes


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


# Plot results
objects = ('Random Forest', 'Naive Bayes', 'Random')
y_pos = np.arange(len(objects))
performance = [accuracy(randomForestPrediction, diagnosisBinary),
               accuracy(gaussianNaiveBayes, diagnosisBinary),
               accuracy(randomPrediction, diagnosisBinary)]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('Accuracy in Cancer Prediction')
 
plt.show()

# Choosing training number
numberVector = []
accuracyVectorRF = []
accuracyVectorNB = []
for i in range(1,14):
    numberVector.append(i*23)
    aRF = accuracy(randomForest.randomForestFunction(i*7, features, diagnosisBinary), diagnosisBinary)
    accuracyVectorRF.append(aRF)
    aNB = accuracy(gaussianNaiveBayes.gaussianBayes(i*7, features, diagnosisBinary), diagnosisBinary)
    accuracyVectorNB.append(aNB)
    
#plt.plot(numberVector, accuracyVector)
plt.plot(numberVector, accuracyVectorRF, 'b-', label='Random Forest')
plt.plot(numberVector, accuracyVectorNB, 'g-', label='Random Naive Bayes')
plt.ylabel("Accuracy")
plt.xlabel("Training Number")
plt.show()
