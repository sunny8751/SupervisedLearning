import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Boosting extends LearningAlgorithm {

    Float c = null;
    Integer m = null;
    
    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("confidenceFactor", .05, .5, .05);
        GridSearchParameters p2 = new GridSearchParameters("minNumObj", 1, 50, 1);
        
        double bestAccuracy = 0;
        String bestParameters = "";
        for (double i = p1.low; i <= p1.high; i+=p1.step) {
            for (double j = p2.low; j <= p2.high; j+=p2.step) {
                J48 tree = ((J48) ((AdaBoostM1) model).getClassifier());
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

    public Boosting(float c, int m) {
        this.c = c;
        this.m = m;
    }

    public Boosting() {
    }
    
    @Override
    protected Classifier getModel() {
        AdaBoostM1 model = new AdaBoostM1();
        J48 inner = new J48();
        model.setClassifier(inner);

        if (c == null || m == null) {
            System.out.println("No model parameters set");
        } else {
            inner.setConfidenceFactor(c);
            inner.setMinNumObj(m);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "boosting";
    }
}