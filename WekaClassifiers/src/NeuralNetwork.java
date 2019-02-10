import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class NeuralNetwork extends LearningAlgorithm {

    Float l = null;
    Float m = null;
    Integer n = null;

    @Override
    protected GridSearchResult gridSearchHelper(Classifier model, String dataset, Instances train, StringBuilder result) throws Exception {
        GridSearchParameters p1 = new GridSearchParameters("learningRate", .1, 1, .1);
        GridSearchParameters p2 = new GridSearchParameters("momentum", .1, 1, .1);
        GridSearchParameters p3 = new GridSearchParameters("trainingTime", 300, 700, 100);

        double bestAccuracy = 0;
        String bestParameters = "";
        for (double i = p1.low; i <= p1.high; i+=p1.step) {
            for (double j = p2.low; j <= p2.high; j+=p2.step) {
                for (int k = (int) p3.low; k <= (int) p3.high; k+= (int) p3.step) {
                    ((MultilayerPerceptron) model).setLearningRate(i);
                    ((MultilayerPerceptron) model).setMomentum(j);
                    ((MultilayerPerceptron) model).setTrainingTime(k);

                    double accuracy = Utils.CVTest(model, train, null);
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestParameters = String.format("%s: %f\t%s: %f\t%s: %d", p1.parameter, i, p2.parameter, j, p3.parameter, k);
                    }

                    result.append(String.format("(%f,%f,%d,%f) ", i, j, k, accuracy));
                }
            }
        }
        return new GridSearchResult(bestAccuracy, bestParameters);
    }

    public NeuralNetwork(float l, float m, int n) {
        this.l = l;
        this.m = m;
        this.n = n;
    }

    public NeuralNetwork() {
    }

    @Override
    protected Classifier getModel() {
        MultilayerPerceptron model = new MultilayerPerceptron();

        if (l == null || m == null || n == null) {
            System.out.println("WARNING: No model parameters set");
        } else {
            model.setLearningRate(l);
            model.setMomentum(m);
            model.setTrainingTime(n);
        }
        return model;
    }

    @Override
    protected String getName() {
        return "neuralNetwork";
    }
}