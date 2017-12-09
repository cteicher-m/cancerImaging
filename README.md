# cancerImaging 
The system can be broken down into three components: <br>
1. Data files:
  * fullData.txt contains the complete dataset
  * simpleData.txt contains data filtered for only binary 0.0 or 1.0 classification
  * features.txt contains feature vectors for each of the entries
2. Algorithm files:
  * base.py (random base case), decisionTrees.py, gaussianNaiveBayes.py, knn\_modified.py, knn.py, naiveBayes.py, randomForest.py, supportVector.py
3. Main testing files:
  * classifier.py
  * tuning.py

Each of the algorithm files contains its respective prototype classification function. Via python's import operator, these are loaded into the main testing file, classifier.py, which then passes the data set as well as any additional arguments to each of the classifiers via python's dot notation (ex: decisionTrees.decisionTreeFunction()). To receive the results (in the form of a matplotlib print out), simply run *python classifier.py* in terminal. For examples of tuning, you can likewise run *python tuning.py*; this will output graphs used during the process of tuning parameters for each classifier.
