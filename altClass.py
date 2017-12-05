import numpy as np 
import csv
import matplotlib.pyplot as plt

# import classifiers
import base # random case
import randomForest


# Load diagnosis data as numpy array
diagnosis = csv.reader(open('alternate.txt'),delimiter = ",")
diagnosisBinary = []
features = []
for row in diagnosis:
    print(row)
    diagnosisBinary.append(float(row[13]))
    features.append([float(row[0]),float(row[1]),float(row[2]),
                     float(row[3]),float(row[4]),float(row[5]),float(row[6]),
                     float(row[7]),float(row[8]),float(row[9]),float(row[10]),
                     float(row[11]),float(row[12])])
        
diagnosisBinary = np.array(diagnosisBinary).astype(np.float)
features = np.array(features).astype(np.float)



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

randomPrediction = base.randomize(150, features, diagnosisBinary)
randomForestPrediction = randomForest.randomForestFunction(150, features, diagnosisBinary)
print(randomPrediction)

# Plot results
objects = ('Random Forest', 'Random')
y_pos = np.arange(len(objects))
performance = [accuracy(randomForestPrediction, diagnosisBinary),  
               accuracy(randomPrediction, diagnosisBinary)]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.title('Accuracy in Tumor Detection')
 
plt.show()