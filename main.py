    # coding=utf-8


from helper_functions import *
#trainingData = []
trainingData = load_data("train")
print(len(trainingData))

#dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
#print(trainingData)
separated = separateByClass(trainingData)
print('Separated instances: ', )
print(len(separated))

summaries = summarizeByClass(trainingData)
predictions = getPredictions(summaries, trainingData)
#print(predictions)
accuracy = getAccuracy(trainingData, predictions)
print("training Data Accuracy ", accuracy)
##############################################3
testData = load_data("test")
#predictions = getPredictions(summaries, testData)
accuracy = getAccuracy(testData, predictions)
print('Test Data Accuracy',accuracy)



#print('Summary ', summary)
#x = 71.5
#mean = 73
#stdev = 6.2
#probability = getPDF(x, mean, stdev)
#print('Probability ',probability)


