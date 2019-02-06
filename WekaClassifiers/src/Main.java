
public class Main {

    /**
     * For each learning algorithm, run grid search first to get the optimal parameters.
     * Then run findScores() with a new instance of the model, passing in the optimal parameters into the constructor,
     * to get the learning curve scores. After getting a results.txt for all the algorithms,
     * run plotAllResults.py to get a graph of the linear curve.
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        String wineDataset = "winequalitywhite";
        String germanDataset = "germancredit";

        /**
         * Decision Trees with Pruning
         */
//        new DecisionTreeWithPruning().gridSearch(wineDataset);
//        new DecisionTreeWithPruning(.2f, 1).findScores(wineDataset);

//        new DecisionTreeWithPruning().gridSearch(germanDataset);
//        new DecisionTreeWithPruning(.2f, 26).findScores(germanDataset);

        /**
         * Neural Networks
         */
//        new NeuralNetwork().gridSearch(wineDataset);
//        new NeuralNetwork(.1f, .7f, 700).findScores(wineDataset);

//        new NeuralNetwork().gridSearch(germanDataset);
//        new NeuralNetwork(.2f, .5f, 300).findScores(germanDataset);

        /**
         * Boosting
         */
//        new Boosting().gridSearch(wineDataset);
//        new Boosting(.15f, 1).findScores(wineDataset);

//        new Boosting().gridSearch(germanDataset);
//        new Boosting(.35f, 27).findScores(germanDataset);

        /**
         * SVM linear kernel
         */
//        new svmLinear().gridSearch(wineDataset);
//        new svmLinear(10).findScores(wineDataset); // c=1000 has very similar scores but a little lower

//        new svmLinear().gridSearch(germanDataset);
//        new svmLinear(100).findScores(germanDataset);

        /**
         * SVM rbf kernel
         */
//        new svmRbf().gridSearch(wineDataset);
//        new svmRbf(10, 100).findScores(wineDataset);
//
//        new svmRbf().gridSearch(germanDataset);
//        new svmRbf(1, 1f).findScores(germanDataset);
//        new svmRbf(1, .1f).findScores(germanDataset); / ** not grid search results?

        /**
         * kNN with uniform weighting
         */
//        new knnUniform().gridSearch(wineDataset);
//        new knnUniform(1).findScores(wineDataset);
//
//        new knnUniform().gridSearch(germanDataset);
//        new knnUniform(13).findScores(germanDataset);

        /**
         * kNN with distance weighting
         */
//        new knnDistance().gridSearch(wineDataset);
//        new knnDistance(29).findScores(wineDataset);
//
//        new knnDistance().gridSearch(germanDataset);
//        new knnDistance(13).findScores(germanDataset);
    }

}
