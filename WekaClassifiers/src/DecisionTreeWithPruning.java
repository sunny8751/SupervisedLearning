import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DecisionTreeWithPruning extends LearningAlgorithm {

    Float c = null;
    Integer m = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train) throws Exception {
//        CVParameterSelection selection = new CVParameterSelection();
//        selection.setNumFolds(5);
//        selection.addCVParameter("C .05 .95 10");
//        selection.addCVParameter("M 1 50 50");
//        selection.buildClassifier(train);
//        selection.getBestClassifierOptions();

        GridSearchParameters p1 = new GridSearchParameters("confidenceFactor", .05, .5, .05);
        GridSearchParameters p2 = new GridSearchParameters("minNumObj", 1, 50, 1);

        double bestAccuracy = 0;
        String bestParameters = "";
        J48 tree = (J48) model;
        for (double i = p1.low; i <= p1.high; i+=p1.step) {
            for (double j = p2.low; j <= p2.high; j+=p2.step) {
                tree.setConfidenceFactor((float) i);
                tree.setMinNumObj((int) j);
                double accuracy = Utils.CVTest(model, train, null);
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestParameters = p1.parameter + ": " + i + "\t" + p2.parameter + ": " + j;
                }
            }
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public DecisionTreeWithPruning(float c, int m) {
        this.c = c;
        this.m = m;
    }

    public DecisionTreeWithPruning() {
    }

    @Override
    protected Classifier getModel() {
        J48 model = new J48();
        model.setUnpruned(false);

        if (c == null || m == null) {
            System.out.println("WARNING: No model parameters set");
        } else {
            model.setConfidenceFactor(c);
            model.setMinNumObj(m);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "decisionTreeWithPruning";
    }
}