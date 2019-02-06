import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;

public class svmLinear extends LearningAlgorithm {

    Float c = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("c", -3, 3, 1);

        double bestAccuracy = 0;
        String bestParameters = "";
        for (int i = (int) p1.low; i <= (int) p1.high; i += (int) p1.step) {
            int c = (int) Math.pow(10, i);
            ((SMO) model).setC(c);
            double accuracy = Utils.CVTest(model, train, null);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestParameters = p1.parameter + ": " + c;
            }
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public svmLinear(float c) {
        this.c = c;
    }

    public svmLinear() {
    }

    @Override
    protected Classifier getModel() {
        SMO model = new SMO();
        PolyKernel polyKernel = new PolyKernel();
        polyKernel.setExponent(1);
        model.setKernel(polyKernel);

        if (c == null) {
            System.out.println("WARNING: No model parameters set");
        } else {
            model.setC(c);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "svmLinear";
    }
}