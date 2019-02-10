import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DecisionTreeUnpruned extends LearningAlgorithm {

    Integer m = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train, StringBuilder result) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("minNumObj", 1, 1, 1);

        double bestAccuracy = 0;
        String bestParameters = "";
        J48 tree = (J48) model;
        for (int i = (int) p1.low; i <= (int) p1.high; i += (int) p1.step) {
            tree.setMinNumObj(i);
            double accuracy = Utils.CVTest(model, train, null);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestParameters = p1.parameter + ": " + i;
            }
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public DecisionTreeUnpruned(int m) {
        this.m = m;
    }

    public DecisionTreeUnpruned() {
    }

    @Override
    protected Classifier getModel() {
        J48 model = new J48();
        model.setUnpruned(true);

        if (m == null) {
            System.out.println("WARNING: No model parameters set");
        } else {
            model.setMinNumObj(m);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "decisionTreeUnpruned";
    }
}