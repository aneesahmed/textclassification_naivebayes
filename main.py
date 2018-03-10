    # coding=utf-8
import numpy as np

from helper_functions import *
#trainingData = []
trainingData = load_data()
#print(trainingData)

#dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
#print(trainingData)
separated = separateByClass(trainingData)
print('Separated instances: ')
print(len(separated))
for key,value in separated.items():
    print("key", key)
#numbers = [1,2,3,4,5]
#print(numbers, mean(numbers), stdev(numbers))
#dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(trainingData)
#print('Summary ', summary)
x = 71.5
mean = 73
stdev = 6.2
probability = getPDF(x, mean, stdev)
print('Probability ',probability)
#print(trainingData)
#print("shape", train0.shape[1])
#mean0 = np.array([train0.shape[1]])
#mean1 = np.array(train1.shape[1])
#print("mean array ", mean0.shape[1])

dataset = [[1,20,0], [2,21,1], [3,22,0]]
summary = getMeanStdVector(dataset)

#print('Attribute summaries: ',dataset, (summary) )
#meanStdVector = getMeanStdVector(trainingData)
#print(meanStdVector)
#train_prediction = np.np.zeros((classLabelCount[0] + classLabelCount[1], 5180))
#loadPrection(train0, mean0, std0, train_prediction)
#loadPrection(train1, mean0, std0, train_prediction)
#print(train_prediction)
#print(mean0)
#getMean(train1, mean1)
    #mean0 = getmean(train0)
#print(train0.shape, train1.shape)
#print(classLabelCount)
