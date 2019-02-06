import matplotlib.pyplot as plt
import os

GERMAN = "germancredit"
WINE = "winequalitywhite"
datasets = [GERMAN, WINE]

BOOSTING = "boosting"
DECISION_TREE_PRUNING = "decisionTreeWithPruning"
KNN_DISTANCE = "knnDistance"
KNN_UNIFORM = "knnUniform"
NEURAL_NETWORK = "neuralNetwork"
SVM_LINEAR = "svmLinear"
SVM_RBF = "svmRbf"
algorithms = [BOOSTING, DECISION_TREE_PRUNING, KNN_DISTANCE, KNN_UNIFORM, NEURAL_NETWORK, SVM_LINEAR, SVM_RBF]

def getTotalInstances(dataset):
    if dataset == GERMAN:
        return 1000
    else:
        return 4898

def getTitle(dataset, algorithm):
    title = ""
    if dataset == GERMAN:
        title = "German Credit "
    else:
        title = "White Wine "

    if algorithm == BOOSTING:
        title += "Boosting "
    elif algorithm == DECISION_TREE_PRUNING:
        title += "Decision Tree Pruning "
    elif algorithm == KNN_DISTANCE:
        title += "kNN Distance "
    elif algorithm == KNN_UNIFORM:
        title += "kNN Uniform "
    elif algorithm == NEURAL_NETWORK:
        title += "Neural Network "
    elif algorithm == SVM_LINEAR:
        title += "SVM Linear Kernel "
    elif algorithm == SVM_RBF:
        title += "SVM RBF Kernel "

    title += "Learning Curve"
    return title

def graph(dataset, algorithm, trainScores, testScores, cvScores):
    title = getTitle(dataset, algorithm)
    totalInstances = getTotalInstances(dataset)
    trainInstances = totalInstances * 7 / 10
    x = []
    for split in [20, 40, 60, 80, 100]:
        trainSize = trainInstances * split / 100
        # print("split:", split, ":", trainSize)
        x.append(trainSize)
    # x = splits

    plt.figure()
    plt.title(title)
    scoreLabels = [(trainScores, "Train Score"), (testScores, "Test Score"), (cvScores, "5-fold CV Score")]
    for scores, label in scoreLabels:
        plt.plot(x, scores, '-', label=label)
    plt.legend()
    plt.xlabel('Training Data Size')
    # plt.xlabel('k')
    # plt.ylabel('Average Accuracy Score (%)')
    plt.ylabel('Accuracy Score (%)')
    # plt.show()

    figureDir = "figures/" + dataset + "/"
    if not os.path.exists(figureDir):
        os.makedirs(figureDir)
    plt.savefig(figureDir + dataset + "_" + algorithm + ".png")

def parseResult(line):
    result = []
    scoresStr = line.split("=")[-1].strip()[1:-1]
    for score in scoresStr.split(","):
        score = float(score.strip())
        result.append(score)
    return result


resultsDir = "WekaClassifiers/results/"
for dataset in datasets:
    fileName = resultsDir + dataset + "/"
    for algorithm in algorithms:
        scores = [[],[],[]]
        with open(fileName + algorithm + "/result.txt", 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                scores[i-1] = parseResult(line)
        graph(dataset, algorithm, *scores)



# totalInstances = 4898
# trainScores = [100.0, 100.0, 100.0, 100.0, 100.0]
# testScores = [51.80394826412525, 56.364874063989106, 56.773315180394825, 61.198093941456776, 62.96800544588155]
# cvScores = [50.51094890510949, 53.61050328227571, 56.68449197860963, 59.27816259569814, 62.758821813939925]
# graph('White Wine Boosting Learning Curve (C=0.15, M=1)', totalInstances, splits, (trainScores, "Train Score"), (testScores, "Test Score"), (cvScores, "5-fold CV Score"))
