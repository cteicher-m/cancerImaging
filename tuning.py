# Tuning parameters
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