# Convert data to binary
import numpy as np 
import csv

file = csv.reader(open('fullData.txt'), delimiter=" ")

# Write to new file
new = open('simpleData.txt','w')

for row in file:
    if row[2] == 'NORM':
        new.write('0\n')
    else:
        new.write('1\n')

new.close()