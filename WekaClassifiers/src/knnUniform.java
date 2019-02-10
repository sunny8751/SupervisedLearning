import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;

public class knnUniform extends LearningAlgorithm {

    Integer k = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train, StringBuilder result) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("k", 1, 50, 1);

        double bestAccuracy = 0;
        String bestParameters = "";
        for (int i = (int) p1.low; i <= (int) p1.high; i += (int) p1.step) {
            ((IBk) model).setKNN(i);
            double accuracy = Utils.CVTest(model, train, null);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestParameters = p1.parameter + ": " + i;
            }

            result.append(String.format("(%d,%f) ", i, accuracy));
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public knnUniform(int k) {
        this.k = k;
    }

    public knnUniform() {
    }

    @Override
    protected Classifier getModel() {
        IBk model = new IBk();
        model.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));

        if (k == null) {
            System.out.println("WARNING: No model parameters set");
        } else {
            model.setKNN(k);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "knnUniform";
    }
}