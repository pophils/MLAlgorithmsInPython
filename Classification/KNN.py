

import numpy as NP
from matplotlib import pyplot as PT
from operator import itemgetter 

class KNN(object):
    """
       Class provides implementation of classification 
       problem via k-Nearest Neighbour Algorithm.
       Given a training data set and a test data, we look for the k most 
       similar training example by computing the Euclidean distance between each data point
       and consequently vote for the label that has the most occurrence among the training data.

       For KNN, there is no training phase with the training data is loaded in to the memory.
       This will make it prone to overfitting. 
       The hyperparameter, K can be tuned to provide the best model btw under and over fitting.
       Memory usage by KNN grows linearly with the size of the training data. It tends to work well
       with substantially sized features.

       KNN complexity is O(n + klogk) i.e the 'n' comes from computing the distances between the
       test vector each items in the training data. 'klogk' comes from sorting the k class labels
    """

    def __init__(self, training, trainingLabels):

        if not isinstance(training, NP.ndarray):
            self.training = NP.array(training)
        else:
            self.training = training

        if not isinstance(trainingLabels, NP.ndarray):
            self.trainingLabels = NP.array(trainingLabels)  
        else:
            self.trainingLabels = trainingLabels

        try:
            self.NoOfTrainingFeatures = training.shape[1]
            self.NoOfTrainingSamples = training.shape[0]

        except Exception as e:
            raise e

        if self.NoOfTrainingSamples != self.trainingLabels.shape[0]:
            raise ValueError("Number of training sample is not the same as the number of labels.")

    def Classify(self, testVector, k=1):

        if not isinstance(testVector, NP.ndarray):
            testVector = NP.array(testVector)
        else:
            testVector = testVector

        noOfTestFeatures = testVector.shape[0]
        
        if self.NoOfTrainingFeatures != noOfTestFeatures:
            raise ValueError("The Number of features in the training and test data is not the same")

        # Repeat test data vertically no of training data times to calc difference
        # O(n) the second param in tile() = (noOfRepetitionAlongY, NoOfRepetitionAlongX)
        diffMat = NP.tile(testVector, (self.NoOfTrainingSamples, 1)) - self.training

        squareDiffMat = diffMat ** 2
        sqDistance = squareDiffMat.sum(axis=1)
        euclideanDist = sqDistance ** 0.5

        sortedDistance = NP.argsort(euclideanDist)

        if self.NoOfTrainingSamples < k:
            k = self.NoOfTrainingSamples

        labelClassCountMap = {}
        for index in range(k):
            label = self.trainingLabels[sortedDistance[index]]
            labelClassCountMap[label] = labelClassCountMap.get(label, 0) + 1
            
        sortedClassCount = sorted(labelClassCountMap.items(), key=itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

    def TestClassifier(self, testVectors, testLabels, k=1):

        if not isinstance(testVectors, NP.ndarray):
            testVectors = NP.array(testVectors)
        else:
            testVectors = testVectors

        if not isinstance(testLabels, NP.ndarray):
            testLabels = NP.array(testLabels)  
        else:
            testLabels = testLabels

        noOfTestData = testVectors.shape[0]
        noOfTestLabels = testLabels.shape[0]         

        if noOfTestData != noOfTestLabels:
            raise ValueError("The number of test data and label are not the same.") 

        errorCount = 0
        for vectorIndex in range(noOfTestData):
            classifierResult = self.Classify(testVectors[vectorIndex, :], k)
            actualLabel = testLabels[vectorIndex]
            print("The classifier result: %s, actual result: %s" % (classifierResult, actualLabel))

            if actualLabel != classifierResult:
                errorCount += 1

        errorRate = errorCount / noOfTestData

        print("The total classification error rate is %s" % errorRate)

    @staticmethod
    def normalize(dataSet, maxValMat, minValMat):
        """ Normalize data if features are measured in different ranges.
            newValue = (oldVal - columnMinVal) / (columnMaxVal - columnMinVal)
        """

        if len(dataSet) < 1:
            raise ValueError("The dataset contains no values.")

        if not isinstance(dataSet, NP.ndarray):
            dataSet = NP.array(dataSet)

        if len(maxValMat) < 1:
            maxValMat = NP.max(dataSet, axis=0)

        if len(minValMat) < 1:
            minValMat = NP.min(dataSet, axis=0)  # axis = 0 find min along y axis i.e column

        noOfDatasetRows = dataSet.shape[0]
        normDataset = dataSet - NP.tile(minValMat, (noOfDatasetRows, 1))
        rangeData = minValMat - maxValMat

        normDataset = normDataset / NP.tile(rangeData, (noOfDatasetRows, 1))

        return normDataset, maxValMat, rangeData