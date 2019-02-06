# Analysis of Various Supervised Learning Classifiers

Weka was used to run 5 different classifiers: Decision tree with pruning, Boosting, Neural Network, kNN (both distance and uniform weighting), and Support Vector Machine (both linear and RBF kernel).

I then analyzed these different classifiers against two databases: [White Wine Quality](http://archive.ics.uci.edu/ml/datasets/Wine) and [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29).

# Setup Instructions
1. Download the White Wine Quality and German Credit datasets
2. Split up each dataset into a train (70%) and test (30%) dataset using sklearn's train_test_split. Name "train.arff" and "test.arff" respectively.
3. Use Weka's ArffViewer to convert csv files to arff format.
4. Run "python3 splitData.py", changing the "dataset" variable to create 20, 40, 60, 80, 100% percentage splits from the train.arff data for each dataset.

# Run instructions
The "WekaClassifiers" folder contains the Java code to run the 5 different classifiers. In the "Main.java" file, the function for each classifier is commented out.
For each classifier and each database:
1. Run the gridSearch() method to find the optimal parameters (printed in the standard output) 
2. Use the outputted parameters by plugging them into the constructor of each classifier class and then running findScores(). This will train the classifier on each of the 5 train percentage splits and find the accuracy scores for the train set, a 5-fold cross validation of the train set, and the test set.

Once findScores() is run for each classifier and database (14 runs), run "python3 plotAllResults.py". Matplotlib will be required as a package. This will create a figures directory at the root level with the learning curve graphs for each of the 14 runs.