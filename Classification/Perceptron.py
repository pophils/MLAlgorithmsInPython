
import numpy as NP


class Perceptron():
    """
    Class provides implementation of the Perceptron Machine Learning Algorithm.
    """

    def __init__(self, epoch, eta, averageWeighted=False):
        """
        :param epoch: The number of runs through the training sets
                      It must be chosen to prevent overfitting or underfitting.
        :param eta: The learning rate of the algorithm i.e the rate at which the weight are updated
        :param averageWeighted: If the averaged perceptron model should be used.
        """

        if not isinstance(epoch, int):
            raise TypeError("The epoch must be an integer.")

        if not (isinstance(eta, int) or isinstance(eta, float)):
            raise TypeError("The epoch must be a digit.")

        if not isinstance(averageWeighted, bool):
            raise TypeError("averageWeighted must be a boolean.")

        self.__epoch = epoch
        self.__eta = eta
        self.__weight = self.__averageWeight = self.__aggregateWeight = self.__trainingSet = self.__labels = self.__error = None
        self.__numberOfTrainingSamples = self.__numOfFeatures = self.__numberOfLabels = 0
        self.__averageWeighted = averageWeighted

    def fit_transform(self, trainingSet, trainingLabels):
        """
        Trains the learning algorithm.
        :param trainingSet: The training sets (numpy array).
        :param trainingLabels: The training labels (numpy array).
        :return: tuple (weight vector, error list)
        """

        if not isinstance(trainingSet, NP.ndarray):
            self.__trainingSet = NP.array(trainingSet)
        else:
            self.__trainingSet = trainingSet

        if not isinstance(trainingLabels, NP.ndarray):
            self.__labels = NP.array(trainingLabels)
        else:
            self.__labels = trainingLabels

        try:
            self.__numberOfTrainingSamples = self.__trainingSet.shape[0]
            self.__numOfFeatures = self.__trainingSet.shape[1]
            self.__numberOfLabels = self.__labels.shape[0]

        except Exception as e:
            raise e

        if self.__numberOfLabels != self.__numberOfTrainingSamples:
            raise ValueError("The number of training labels and set are not the same.")

        # init all weights and bias.from
        self.__weight = self.__aggregateWeight = NP.zeros(self.__numOfFeatures + 1)  # the increment is for the bias term
        self.__trainingSet = NP.concatenate((self.__trainingSet, NP.ones((self.__numberOfTrainingSamples, 1))), axis=1)

        for run in range(self.__epoch):
            for cursor in range(self.__numberOfTrainingSamples):
                prediction = Perceptron.__unitStep(NP.dot(self.__weight, self.__trainingSet[cursor, :]))

                if prediction != self.__labels[cursor]:
                    error = self.__labels[cursor] - prediction

                    if self.__error is None:
                        self.__error = []
                    self.__error.append(error)

                    # self.__aggregateWeight += self.__weight this will cause overflow runtime error
                    self.__aggregateWeight = self.__aggregateWeight + self.__weight
                    self.__weight += self.__eta * self.__trainingSet[cursor, :] * error

        if self.__averageWeighted:
            self.__weight = self.__aggregateWeight / (self.__epoch * self.__numberOfTrainingSamples)

        return self.__weight, self.__error

    def predict(self, testSet):
        """
        Run the learning algorithm to the test vector(s).
        :param testSet: The test sets.(numpy array)
        :return: prediction
        """

        if not isinstance(testSet, NP.ndarray):
            testSet = NP.array(testSet)

        try:
            numberOfTestSamples = testSet.shape[0]
            numOfTestFeatures = testSet.shape[1]

        except Exception as e:
            raise e

        if self.__numOfFeatures != numOfTestFeatures:
            raise ValueError("The number of training and test features are not the same.")

        # add bias to test vectors
        testSet = NP.concatenate((testSet, NP.ones((numberOfTestSamples, 1))), axis=1)

        predictions = NP.zeros((numberOfTestSamples,), dtype=int)

        for cursor in range(numberOfTestSamples):
                predictions[cursor] = Perceptron.__unitStep(NP.dot(self.__weight, testSet[cursor, :]))

        return predictions

    @staticmethod
    def __unitStep(weightedSum):
        """
        Makes a new prediction using the Heaviside function
        :param weightedSum: The weighted sum.
        :return: 0 or 1
        """

        return 1 if NP.sign(weightedSum) > 0 else 0

    def test(self, testSet):
        """
        Trains the learning algorithm.
        :param testSet: The test sets (numpy array).
        :return: prediction list
        """

        if not isinstance(testSet, NP.ndarray):
            testSet = NP.array(testSet)

        try:
            numberOfTestSamples = testSet.shape[0]
            numOfTestFeatures = testSet.shape[1]

        except Exception as e:
            raise e

        if self.__numOfFeatures != numOfTestFeatures:
            raise ValueError("There is a mismatch between the number of training and test features.")

        # init all weights and bias.from
        testSet = NP.concatenate((testSet, NP.ones((numberOfTestSamples, 1))), axis=1)

        predictionList = []
        for cursor in range(numberOfTestSamples):
                predictionList.append(Perceptron.__unitStep(NP.dot(self.__weight, testSet[cursor, :])))

        return predictionList