dataset = "winequalitywhite"
# dataset = "germancredit"

dataFilepath = "./WekaClassifiers/datasets/" + dataset + "/train.arff"
splitDir = "./WekaClassifiers/datasets/" + dataset + "/splits/"
headerExists = True
splitPercentages = [20,40,60,80,100]
data = []

# dataFilepath = "./datasets/wine/trainSplit.csv"
# splitDir = "./datasets/wine"
# splitPercentage = 70


header = None
attributeLines = None
with open(dataFilepath, 'r') as f:
    if ".arff" in dataFilepath:
        attributeLines = []
        arffData = False
        headerExists = False
    for line in f:
        if attributeLines != None and (line == "\n" or not arffData):
            if "@data" in line: arffData = True
            attributeLines.append(line)
            continue
        if headerExists:
            header = line
            headerExists = False
            continue
        data.append(line)

fileType = ".csv"
if attributeLines: fileType = ".arff"

import random
random.shuffle(data)
print("total:", len(data))
for percentage in splitPercentages:
    split = data[:int(len(data) * percentage / 100)]
    print(str(percentage) + ":", len(split))

    with open(splitDir+str(percentage)+fileType, 'w') as f:
        if attributeLines: f.writelines(attributeLines)
        if header: f.write(header)
        f.writelines(split)
# splitTrain = data[:int(len(data) * splitPercentage / 100)]
# splitTest = data[int(len(data) * splitPercentage / 100):]
# with open(splitDir+"trainSplit.csv", 'w') as f:
#     if header: f.write(header)
#     f.writelines(splitTrain)
# with open(splitDir+"testSplit.csv", 'w') as f:
#     if header: f.write(header)
#     f.writelines(splitTest)
