

import numpy as NP


class Ridge():
    """
    Class implements Ridge regression. Its a weight regularization / shrinkage
    method of reducing variance (Overfitting in the training data).
    This produce a model that can generalize more than the Ordinary Least Square
    Method.
    """

    def __init__(self, alpha=0.1, normalize=False):
        self.__alpha = alpha
        self.__beta = None
        self.__normalize = normalize
        self.__trainingSet = None
        self.__trainingLabel = None
        self.__noOfTrainingSamples = self.__noOfTrainingFeatures = 0

    def predict(self, testSet):
        """
        Predicts the explanatory variable of a tet set.
        :param testSet:
        :return: prediction numpy array.
        """

        if not isinstance(testSet, NP.ndarray):
            testSet = NP.array(testSet)
        else:
            testSet = testSet

        # Edge cases
        try:
            noOfTestFeatures = testSet.shape[1]
        except Exception as e:
            raise e

        if self.__noOfTrainingFeatures != noOfTestFeatures:
            raise ValueError("The number of training and test features are not the same.")

        testSet = NP.mat(testSet)

        return (testSet * self.__beta).A.flatten()

    def fit(self, trainSet, trainLabel):
        """
        Fits a regression hyperplane along the training data
        :param trainSet:
        :param trainLabel:
        :return: beta, the weight coefficients.
        """

        if not isinstance(trainSet, NP.ndarray):
            self.__trainingSet = NP.array(trainSet)
        else:
            self.__trainingSet = trainSet

        if not isinstance(trainLabel, NP.ndarray):
            self.__trainingLabel = NP.array(trainLabel)
        else:
            self.__trainingLabel = trainLabel

        # Edge cases
        try:
            self.__noOfTrainingSamples = self.__trainingSet.shape[0]
            self.__noOfTrainingFeatures = self.__trainingSet.shape[1]
        except Exception as e:
            raise e

        if self.__noOfTrainingSamples != self.__trainingLabel.shape[0]:
            raise ValueError("The number of training sample and labels are not the same.")

        self.__trainingSet = NP.mat(self.__trainingSet)
        self.__trainingLabel = NP.mat(self.__trainingLabel).T

        # standardized training set -- centralize training labels.
        # Centralizing is done on the labels since they will sure be on the same scale.

        xMean = xVar = 0
        if self.__normalize:
            xMean = NP.mean(self.__trainingSet, axis=0)
            yMean = NP.mean(self.__trainingLabel, axis=0)
            xVar = NP.var(self.__trainingSet, axis=0)

            self.__trainingSet = (self.__trainingSet - xMean) / xVar
            self.__trainingLabel = self.__trainingLabel - yMean

        xTx = self.__trainingSet.T * self.__trainingSet
        L2BiasFactor = NP.eye(self.__trainingSet.shape[1]) * self.__alpha

        invertibleMatrix = xTx + L2BiasFactor

        # Edge case: if alpha is zero, Ridge becomes a normal OLS regression. Hence
        # we need to check if the invertibleMatrix is non singular before moving on.

        if self.__alpha == 0:
            if NP.linalg.det(invertibleMatrix) == 0:
                raise ValueError("The inverse of a singular matrix cannot be found.")

        self.__beta = invertibleMatrix.I * (self.__trainingSet.T * self.__trainingLabel)

        if self.__normalize:
            self.__beta = (self.__beta.T.A + xMean.A) * xVar.A

        return self.__beta.flatten()

    def rmse(self, predictedLabels, actualLabels):
        """
        Returns the root mean square error associated with a prediction.
        :param predictedLabels: predicted values.
        :param actualLabels: actual values.
        """

        if not isinstance(predictedLabels, NP.ndarray):
            predictedLabels = NP.array(predictedLabels)

        if not isinstance(actualLabels, NP.ndarray):
            actualLabels = NP.array(actualLabels)

        # Edge cases
        if predictedLabels.shape[0] == 0:
            raise ValueError("The predicted labels has no data values")

        if actualLabels.shape[0] == 0:
            raise ValueError("The actual labels has no data values")

        if predictedLabels.shape[0] != actualLabels.shape[0]:
            raise ValueError("The number of data values in the predicted and actual labels are not the same")

        tse = ((predictedLabels - actualLabels) ** 2).sum()
        mse = tse / predictedLabels.shape[0]

        return NP.sqrt(mse)


from sklearn.datasets import load_boston
if __name__ == '__main__':

    datasets = load_boston()

    data = datasets.data
    target = datasets.target

    trainSet = data[:450, :]
    trainLabel = target[:450]

    testSet = data[451:, :]
    testLabel = target[451:]

    model = Ridge(alpha=0.21)
    model.fit(trainSet, trainLabel)

    predictions = model.predict(testSet)

    print(predictions)
    print(model.rmse(predictions, testLabel))

    print(predictions.shape)


