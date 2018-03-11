# coding=utf-8
import numpy as np
import csv
import collections as cl
import math
from scipy.stats import norm
#################333


####################
def load_data(filePrefix):
    labels = np.loadtxt("data/" + filePrefix + "_labels.txt", dtype=int)
    lines = csv.reader(open("data/"+filePrefix+".csv",  "rt"))
    dataset = list(lines)
    counter = 0
    row = []
    for i in range(len(dataset)):
        row =  [float(x) +1 for x in dataset[i]]
        #for i in range(len(row)-1):
        #    row[i] = row[i] + 1

        #testVector(row)
        #### adding one to each value
        #row = [(x +1) for x in row]
        #print("updated")
        #testVector(row)
        row.append(labels[counter])
        dataset[i] = row
        counter = counter + 1
        #testVector((dataset[i]))
    return dataset
    #cnt = cl.Counter(labels)
    #print(cnt)
    #print(labels)
######################
def testVector(vector):
    counter = 0
    stop = 15
    #print(len(vector))
    for e in vector:
        if e == 0.0:
            print(counter, e, end=" , ")


            #if stop < counter:
        #    break
######################3333333333333
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
    mn = sum(numbers) / float(len(numbers))
    #counter = 0
    #if mn == 0.0:
    #    print ("zero mean", counter)
    return mn


##############
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    mn = math.sqrt(variance)
    if mn == 0.0:
        mn = 0.001
    return mn

    return mn

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
def getPDF(x, mean, std):

    # calling from getClassProbabilities
    exp = 0
    '''
    if int(x)  == 0:
        x = 1.0
    if int(mean)  == 0:
        mean = 1.0
    if int(std)  == 0:
        std = 0.001
    #print('[',x, mean, std,'],')
    '''
    exp = norm.pdf(x, mean, std)
    if exp < 0.1:
        exp = 0.1
    #print("exp", exp)
    # a = math.pow(x - mean, 2.0)
    #exp = math.exp(-(math.pow(x - mean, 2.0) / (2.0 * math.pow(stdev, 2.0))))

    return exp
    #return (1.0 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#################################33
def getClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += math.log2(getPDF(x, mean, stdev))
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
####################33

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
##########################333
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
