import numpy as np 
import csv

# Load diagnosis data as numpy array
diagnosis = csv.reader(open('simpleData.txt'),delimiter = " ")
data_y = []
for row in diagnosis:
    data_y.append(row)
data_y = np.array(data_y).astype(np.float)
# print data_y.shape,data_y --> 322 entries, values 0 or 1

# Load feature data as numpy array
data_X = np.loadtxt('features.txt')
# print data_X, data_X.shape --> 322 entries, 6 features
