# coding=utf-8
import numpy as np
import csv
import collections as cl
import math
#################333
def load_data():
    labels = np.loadtxt("data/train_labels.txt", dtype=int)

    lines = csv.reader(open("data/train.csv",  "rt"))

    dataset = list(lines)
    counter = 0
    row = []
    for i in range(len(dataset)):
        row =  [float(x) for x in dataset[i]]
        row.append(labels[counter])
        dataset[i] = row
        counter = counter + 1
    return dataset
    #cnt = cl.Counter(labels)
    #print(cnt)
    #print(labels)
######################
def separateByClass(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    #print(separated[1])
    #print(len(separated.keys()))
    return separated
##################
def mean(numbers):
    return sum(numbers) / float(len(numbers))

##############
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

####################
def summarize(data):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
    del summaries[-1]
    return summaries

############
def summarizeByClass(data):
    separated = separateByClass(data)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

###############################333
def getPDF(x, mean, stdev):
    # getClassProbabilities
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#################################33
def getClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= getPDF(x, mean, stdev)
    return probabilities
##############################33
def predict(summaries, inputVector):
	probabilities = getClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

################
######### discardable
def getMeanStdVector(trainingMatrix):
    print(len(trainingMatrix))
    meanStdVector= [(mean(attribute), stdev(attribute)) for attribute in zip(*trainingMatrix)]
    #meanVector = np.mean(matrix, axis=1)
    #meanVector = meanVector.reshape(meanVector.shape[0], 1) # convert 1d rowwise array to column array
    return meanStdVector

###################
def getStdVector(matrix):
    stdVector = np.std(matrix, axis = 1)
    return stdVector
###################
def loadPrection(dataMatrix, predictionVector):
    index = 0
    for row in dataMatrix:
        prediction =1
        #one =  logPdf(row)
        #zero =
        prediction[index] =logPdf()
        index = index + 1

########################3
def logPdf(dataVector, meanVector, stdVector):
    p = 0.0
    for i in range(dataVector.shape):
         p = p + math.log2(pdf(dataVector[0], meanVector[0], stdVector[0]))
    return p
################
def pdf(x, mean, std):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
##########33
