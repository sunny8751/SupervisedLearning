import matplotlib.pyplot as plt
import numpy as np

def graphAccuracies():
    title = "Accuracies of Learning Algorithms"
    algorithms = ['Decision Tree', 'Boosting', 'SVM', 'Neural Network', 'kNN']
    wineAccuracies = [54.32, 63.9, 64.4, 54.3, 64.06]
    germanAccuracies = [71.67, 73, 73, 69.7, 72.67]

    for i, x in enumerate(wineAccuracies):
        wineAccuracies[i] -= 50

    for i, x in enumerate(germanAccuracies):
        germanAccuracies[i] -= 50

    # data to plot
    n_groups = len(algorithms)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, wineAccuracies, bar_width,
                     alpha=opacity,
                     color='b',
                     label='White Wine',
                     bottom=50)

    rects2 = plt.bar(index + bar_width, germanAccuracies, bar_width,
                     alpha=opacity,
                     color='g',
                     label='German Credit',
                     bottom=50)

    plt.xlabel('Learning Algorithms')
    plt.ylabel('Accuracy Score (%)')
    plt.title(title)
    plt.xticks(index + bar_width, algorithms)
    plt.legend(loc=4)

    plt.show()

    plt.savefig("figures/algorithms.png")

def graphRuntimes():
    title = "Runtimes of Learning Algorithms"
    algorithms = ['Decision Tree', 'Boosting', 'SVM', 'Neural Network', 'kNN']
    wineRuntimes = [.149, 2.07, 28.79, 11.8, .51]
    germanRuntimes = [.01, .02, 11.48, 1.83, .03]

    # data to plot
    n_groups = len(algorithms)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, wineRuntimes, bar_width,
                     alpha=opacity,
                     color='b',
                     label='White Wine')

    rects2 = plt.bar(index + bar_width, germanRuntimes, bar_width,
                     alpha=opacity,
                     color='g',
                     label='German Credit')

    plt.xlabel('Learning Algorithms')
    plt.ylabel('Runtime (seconds)')
    plt.title(title)
    plt.xticks(index + bar_width, algorithms)
    plt.legend()

    plt.show()

    # plt.savefig("figures/runtimes.png")

# graphAccuracies()
graphRuntimes()
