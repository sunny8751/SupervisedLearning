import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileWriter;

public abstract class LearningAlgorithm {

    protected class GridSearchParameters {
        public String parameter;
        public double low, high, step;

        public GridSearchParameters(String _parameter, double _low, double _high, double _step) {
            parameter = _parameter;
            low = _low;
            high = _high;
            step = _step;
        }
    }

    protected class GridSearchResult {
        public double bestAccuracy = 0;
        public String bestParameters = "";

        public GridSearchResult(double _bestAccuracy, String _bestParameters) {
            bestAccuracy = _bestAccuracy;
            bestParameters = _bestParameters;
        }
    }

    protected abstract Classifier getModel();

    protected abstract String getName();

    protected abstract GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train)
            throws Exception;

    public void findScores(String dataset) throws Exception {
        Instances test = Utils.readDataFile("datasets/" + dataset + "/test.arff");
        Utils.findScores(getModel(), test, dataset, getName());
    }

    public void gridSearch(String dataset) throws Exception {
        Classifier model = getModel();
        Instances train = Utils.readDataFile("datasets/" + dataset + "/splits/100.arff");
        GridSearchResult gridSearchResult = gridSearchHelper(model, dataset, train);

        StringBuilder result = new StringBuilder();

        result.append("Grid search results:\n");
        result.append("Accuracy: " + gridSearchResult.bestAccuracy + "\n");
        result.append(gridSearchResult.bestParameters + "\n");

        String resultStr = result.toString();
        System.out.print(resultStr);

        BufferedWriter writer = new BufferedWriter(new FileWriter(Utils.getDir(dataset, getName()) + "gridsearch"));
        writer.write(resultStr);
        writer.close();
    }
}
