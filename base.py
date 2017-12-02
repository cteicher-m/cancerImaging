# Random Base Case
import random

def randomize(trainNumber, features, groundTruth):
    predictions = {}
    for i in range(trainNumber, 322):
        n = random.random()
        # if n < 115/322:
        if n < 0.5:
            predictions[i] = 1.0
        else:
            predictions[i] = 0.0
    return predictions