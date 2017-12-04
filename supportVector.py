# Support Vector Machine Classification

# Naive Bayes implementation
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy as np

def naiveBayesAlg(trainNumber, features, groundTruth):
    
    def splitDataset(dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))
        return [trainSet, copy]

    def separateByClass(dataset):
        separated = {}
        for i in range(len(dataset)):
            print(dataset)
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    def mean(numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(numbers):
        avg = mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

    def summarize(dataset):
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries

    def summarizeByClass(dataset):
        separated = separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.iteritems():
            summaries[classValue] = summarize(instances)
        return summaries

    def calculateProbability(x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def calculateClassProbabilities(summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                print(classSummaries[i])
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= calculateProbability(x, mean, stdev)
        return probabilities

    def predict(summaries, inputVector):
        probabilities = calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(summaries, testSet):
        predictions = []
        for i in range(len(testSet)):
            result = predict(summaries, testSet[i])
            predictions.append(result)
        return predictions

    def getAccuracy(testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0

    def main(trainNumber, features, groundTruth):
        
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
        
        trainingSet = {"0.0": [], "1.0": []}
        testSet = {"0.0": [], "1.0": []}
        
        for i in range (0,trainNumber):
            if groundTruth[i] == 0.0:
                trainingSet["0.0"].append(fullFeatures[i])
            else:
                trainingSet["1.0"].append(fullFeatures[i])
                
        for i in range (trainNumber,322):
            if gT[i-trainNumber] == 0.0:
                testSet["0.0"].append(fullFeatures[i])
            else:
                testSet["1.0"].append(fullFeatures[i])
                # prepare model
        # summaries = summarizeByClass(trainingSet)
        # test model
        predictions = getPredictions(trainingSet, f)
        print(predictions)
        #accuracy = getAccuracy(testSet, predictions)
        #print('Accuracy: {0}%').format(accuracy)

    main(trainNumber, features, groundTruth)