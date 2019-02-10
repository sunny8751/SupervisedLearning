import matplotlib.pyplot as plt
import os

GERMAN = "germancredit"
WINE = "winequalitywhite"
datasets = [WINE, GERMAN]

BOOSTING = "boosting"
DECISION_TREE_PRUNING = "decisionTreeWithPruning"
DECISION_TREE_UNPRUNED = "decisionTreeUnpruned"
KNN_DISTANCE = "knnDistance"
KNN_UNIFORM = "knnUniform"
NEURAL_NETWORK = "neuralNetwork"
SVM_LINEAR = "svmLinear"
SVM_RBF = "svmRbf"
algorithms = [BOOSTING, DECISION_TREE_PRUNING, DECISION_TREE_UNPRUNED, KNN_DISTANCE, KNN_UNIFORM, NEURAL_NETWORK, SVM_LINEAR, SVM_RBF]

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
    elif algorithm == DECISION_TREE_UNPRUNED:
        title += "Decision Tree Unpruned "
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
    else:
        title += algorithm

    title += "Learning Curve"
    return title

def graph(dataset, algorithm, scoreLabels, fileName=None, x=None, xLabel="Training Data Size"):
    title = getTitle(dataset, algorithm)
    totalInstances = getTotalInstances(dataset)
    trainInstances = totalInstances * 7 / 10
    if x == None:
        x = []
        for split in [20, 40, 60, 80, 100]:
            trainSize = trainInstances * split / 100
            # print("split:", split, ":", trainSize)
            x.append(trainSize)
    # x = splits

    plt.figure()
    plt.title(title)
    for scores, label in scoreLabels:
        if label:
            plt.plot(x, scores, '-', label=label)
        else:
            plt.plot(x, scores, '-')
    plt.legend()
    plt.xlabel(xLabel)
    # plt.xlabel('k')
    # plt.ylabel('Average Accuracy Score (%)')
    plt.ylabel('Accuracy Score (%)')
    # plt.show()

    figureDir = "figures/" + dataset + "/"
    if not os.path.exists(figureDir):
        os.makedirs(figureDir)
    if fileName == None:
        plt.savefig(figureDir + dataset + "_" + algorithm + ".png")
    else:
        plt.savefig(figureDir + fileName)

def parseResult(line):
    result = []
    scoresStr = line.split("=")[-1].strip()[1:-1]
    for score in scoresStr.split(","):
        score = float(score.strip())
        result.append(score)
    return result

def parseGridSearch(fileName):
    with open(fileName, 'r') as f:
        tokens = f.readlines()[1].strip().split(" ")
        x = [int(float(token.split(",")[0][1:])) for token in tokens]
        scores = [float(token.split(",")[1][:-1]) for token in tokens]
    return x, scores


resultsDir = "WekaClassifiers/results/"
for dataset in datasets:
    fileName = resultsDir + dataset + "/"
    for algorithm in algorithms:
        scores = [[], [], []]
        with open(fileName + algorithm + "/result.txt", 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                scores[i-1] = parseResult(line)
        scoreLabels = [(scores[0], "Train Score"), (scores[1], "Test Score"), (scores[2], "5-fold CV Score")]
        graph(dataset, algorithm, scoreLabels)

# graph grid search for some algorithms
for dataset in datasets:
    fileName = resultsDir + dataset + "/"

    # kNN
    scores = [[], []]
    x = []
    for i, algorithm in enumerate([KNN_UNIFORM, KNN_DISTANCE]):
        x, scores[i] = parseGridSearch(fileName + algorithm + "/gridsearch")
        # with open(fileName + algorithm + "/gridsearch", 'r') as f:
        #     tokens = f.readlines()[1].strip().split(" ")
        #     scores[i] = [float(token.split(",")[1][:-1]) for token in tokens]
    scoreLabels = [(scores[0], "Uniform Weight"), (scores[1], "Distance Weight")]
    graph(dataset, "kNN Grid Search ", scoreLabels, fileName="k_gridsearch.png", x=x, xLabel="K")

    # boosting
    with open(fileName + BOOSTING + "/gridsearch", 'r') as f:
        tokens = f.readlines()[1].strip().split(" ")
        bestScore = 0
        best = []
        for token in tokens:
            token = token[1:-1]
            nums = token.split(",")
            p1 = float(nums[0])
            p2 = float(nums[1])
            p3 = float(nums[2])
            score = float(nums[3])
            if score > bestScore:
                bestScore = score
                best = [p1, p2, p3]
        print(best)
    # x, scores = parseGridSearch(fileName + BOOSTING + "/gridsearch")
    # scoreLabels = [(scores, "5-fold CV Score")]
    # graph(dataset, "Boosting Grid Search ", scoreLabels, fileName="boosting_gridsearch.png", x=x, xLabel="Number of Iterations")

    # # svm rbf
    # x, scores = parseGridSearch(fileName + SVM_RBF + "/gridsearch")
    # scoreLabels = [(scores, "5-fold CV Score")]
    # graph(dataset, "SVM RBF Kernel Grid Search ", scoreLabels, fileName="svm_rbf_gridsearch.png", x=x, xLabel="")

# totalInstances = 4898
# trainScores = [100.0, 100.0, 100.0, 100.0, 100.0]
# testScores = [51.80394826412525, 56.364874063989106, 56.773315180394825, 61.198093941456776, 62.96800544588155]
# cvScores = [50.51094890510949, 53.61050328227571, 56.68449197860963, 59.27816259569814, 62.758821813939925]
# graph('White Wine Boosting Learning Curve (C=0.15, M=1)', totalInstances, splits, (trainScores, "Train Score"), (testScores, "Test Score"), (cvScores, "5-fold CV Score"))
