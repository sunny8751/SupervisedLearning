# Analysis of Various Supervised Learning Classifiers

Weka was used to run 5 different classifiers: Decision tree with pruning, Boosting, Neural Network, kNN (both distance and uniform weighting), and Support Vector Machine (both linear and RBF kernel).

I then analyzed these different classifiers against two databases: [White Wine Quality](http://archive.ics.uci.edu/ml/datasets/Wine) and [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29).

# Run instructions
The "WekaClassifiers" folder contains the Java code to run the 5 different classifiers. In the "Main.java" file, the function for each classifier is commented out.
For each classifier and each database:
1. Run the gridSearch() method to find the optimal parameters (printed in the standard output) 
2. Use the outputted parameters by plugging them into the constructor of each classifier class and then running findScores().

Once findScores() is run for each classifier and database (14 runs), run "python3 plotAllResults.py". Matplotlib will be required as a package. This will create a figures directory at the root level with the learning curve graphs for each of the 14 runs.