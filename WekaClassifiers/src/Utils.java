import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;

public class Utils {
    public static Instances readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }
        Instances data = null;
        try {
            data = new Instances(inputReader);
        } catch (IOException e) {
            e.printStackTrace();
        }
        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);

        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static double calculateAccuracy(Classifier model, ArrayList<Prediction> predictions, String fileName) throws IOException {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.get(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        double accuracy = 100 * correct / predictions.size();

        if (fileName == null || fileName.equals("")) {
            return accuracy;
        }

        // Print current classifier's name and accuracy in a complicated,
        // but nice-looking way.
//        System.out.println("Accuracy of " + model.getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy)
//                + "\n---------------------------------");

        // write to file in results folder
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        writer.write(model.toString());
        writer.close();

        // Uncomment to see the summary for each training-testing pair.
//        System.out.println(model.toString());
        return accuracy;
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }

    public static double TrainTest(Classifier model, Instances train, String fileName) throws Exception {
//        System.out.println("Train results:");
        // Collect every group of predictions for current model
        ArrayList<Prediction> predictions = new ArrayList<Prediction>();

        Evaluation validation = classify(model, train, train);

        predictions.addAll(validation.predictions());

        // Uncomment to see the summary for each training-testing pair.
        // System.out.println(model.toString());

        // Calculate overall accuracy of current classifier on all splits
        return calculateAccuracy(model, predictions, fileName);
    }

    public static double TestTest(Classifier model, Instances train, Instances test, String fileName) throws Exception {
//        System.out.println("Test results:");
        // Collect every group of predictions for current model
        ArrayList<Prediction> predictions = new ArrayList<Prediction>();

        Evaluation validation = classify(model, train, test);

        predictions.addAll(validation.predictions());

        // Uncomment to see the summary for each training-testing pair.
        // System.out.println(model.toString());

        // Calculate overall accuracy of current classifier on all splits
        return calculateAccuracy(model, predictions, fileName);
    }

    public static double CVTest(Classifier model, Instances train, String fileName) throws Exception {
//        if (fileName != null && !fileName.equals("")) {
//            System.out.println("CV-5 results:");
//        }
        // Collect every group of predictions for current model
        ArrayList<Prediction> predictions = new ArrayList<Prediction>();

        // Do 5-split cross validation
        Instances[][] split = crossValidationSplit(train, 5);

        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];

        // For each training-testing split pair, train and test the classifier
        for (int i = 0; i < trainingSplits.length; i++) {
            Evaluation validation = classify(model, trainingSplits[i], testingSplits[i]);

            predictions.addAll(validation.predictions());
        }

        // Calculate overall accuracy of current classifier on all splits
        return calculateAccuracy(model, predictions, fileName);
    }

    public static void runTests(Classifier model, Instances test, int split, String dataset, String dir, StringBuilder result, ArrayList<Double> trainScores, ArrayList<Double> testScores, ArrayList<Double> cvScores) throws Exception {
        Instances train = readDataFile("datasets/" + dataset + "/splits/" + split + ".arff");

        trainScores.add(TrainTest(model, train, dir + split + " train"));

        long startTime = 0;
        if (split == 100) {
            startTime = System.currentTimeMillis();
        }
        testScores.add(TestTest(model, train, test, dir + split + " test"));
        if (split == 100) {
            String timeResult = "Time to run: " + ((System.currentTimeMillis() - startTime) / 1000f) + " seconds" + "\n";
            result.append(timeResult);
        }
        cvScores.add(CVTest(model, train, dir + split + " cv"));
    }

    private static String getScoresString(ArrayList<Double> scores) {
        StringBuilder builder = new StringBuilder();
        builder.append("[");
        for (int i = 0; i < scores.size(); i++) {
            double score = scores.get(i);
            builder.append(score + ((i == scores.size() - 1) ? "" : ", "));
        }
        builder.append("]");
        return builder.toString();
    }

    public static String getDir(String dataset, String modelName) {
        return "results/" + dataset + "/" + modelName + "/";
    }

    public static void findScores(Classifier model, Instances test, String dataset, String modelName) throws Exception {
        String dir = Utils.getDir(dataset, modelName);
        int[] splits = new int[]{20, 40, 60, 80, 100};

        ArrayList<Double> trainScores = new ArrayList<Double>();
        ArrayList<Double> testScores = new ArrayList<Double>();
        ArrayList<Double> cvScores = new ArrayList<Double>();

        StringBuilder result = new StringBuilder();
        for (int split : splits) {
//          System.out.println("==================================");
//          System.out.println("Split: " + split);
            runTests(model, test, split, dataset, dir, result, trainScores, testScores, cvScores);
        }

        result.append("trainScores = " + getScoresString(trainScores) + "\n");
        result.append("testScores = " + getScoresString(testScores) + "\n");
        result.append("cvScores = " + getScoresString(cvScores) + "\n");
        String resultStr = result.toString();

        System.out.print(resultStr);

        BufferedWriter writer = new BufferedWriter(new FileWriter(dir + "result.txt"));
        writer.write(resultStr);
        writer.close();
    }
}
