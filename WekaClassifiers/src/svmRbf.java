import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class svmRbf extends LearningAlgorithm {

    Float c = null;
    Float gamma = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("c", -3, 3, 1);
        GridSearchParameters p2 = new GridSearchParameters("gamma", -3, 3, 1);

        double bestAccuracy = 0;
        String bestParameters = "";
        for (int i = (int) p1.low; i <= (int) p1.high; i += (int) p1.step) {
            for (int j = (int) p2.low; j <= (int) p2.high; j += (int) p2.step) {
                int c = (int) Math.pow(10, i);
                int gamma = (int) Math.pow(10, j);
                ((SMO) model).setC(c);
                ((RBFKernel) ((SMO) model).getKernel()).setGamma(gamma);

                double accuracy = Utils.CVTest(model, train, null);
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestParameters = String.format("%s: %d\t%s: %d", p1.parameter, c, p2.parameter, gamma);
                }
            }
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public svmRbf(float c, float gamma) {
        this.c = c;
        this.gamma = gamma;
    }

    public svmRbf() {
    }

    @Override
    protected Classifier getModel() {
        SMO model = new SMO();
        RBFKernel rbfKernel = new RBFKernel();
        model.setKernel(rbfKernel);

        if (c == null || gamma == null) {
            System.out.println("WARNING: No model parameters set");
        } else {
            model.setC(c);
            rbfKernel.setGamma(gamma);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "svmRbf";
    }
}