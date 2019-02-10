import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Boosting extends LearningAlgorithm {

    Float c = null;
    Integer m = null;
    Integer n = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train, StringBuilder result) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("numIterations", 1, 15, 2);
        GridSearchParameters p2 = new GridSearchParameters("confidenceFactor", .05, .45, .1);
        GridSearchParameters p3 = new GridSearchParameters("minNumObj", 1, 29, 2);

        double bestAccuracy = 0;
        String bestParameters = "";
        for (double i = p1.low; i <= p1.high; i += p1.step) {
            for (double j = p2.low; j <= p2.high; j += p2.step) {
                for (double k = p3.low; k <= p3.high; k += p3.step) {
                    ((AdaBoostM1) model).setNumIterations((int) i);
                    ((J48) ((AdaBoostM1) model).getClassifier()).setConfidenceFactor((float) j);
                    ((J48) ((AdaBoostM1) model).getClassifier()).setMinNumObj((int) k);
                    double accuracy = Utils.CVTest(model, train, null);
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestParameters = String.format("%s: %d, %s: %f, %s: %d", p1.parameter, (int) i, p2.parameter, j, p3.parameter, (int) k);;
                    }

                    result.append(String.format("(%d,%f,%d,%f) ", (int) i, j, (int) k, accuracy));
                }
            }
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public Boosting(Float c, Integer m, Integer n) {
        this.c = c;
        this.m = m;
        this.n = n;
    }

    public Boosting() {
    }

    @Override
    protected Classifier getModel() {
        AdaBoostM1 model = new AdaBoostM1();
        J48 inner = new J48();
        model.setClassifier(inner);

        if (c == null || m == null || n == null) {
            System.out.println("No model parameters set");
        } else {
            inner.setConfidenceFactor(c);
            inner.setMinNumObj(m);
            model.setNumIterations(n);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "boosting";
    }
}