import warnings
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import random

warnings.simplefilter('ignore')
matplotlib.use('TkAgg')


class MyBGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, batch_size=32, learning_rate=0.001, no_epochs=1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]

        for epoch in range(no_epochs):
            indices = list(range(len(x)))
            random.shuffle(indices)

            for i in range(0, len(x), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_x = [x[idx] for idx in batch_indices]
                batch_y = [y[idx] for idx in batch_indices]

                gradients = self.compute_gradients(batch_x, batch_y)

                for j in range(len(x[0])):
                    self.coef_[j] -= learning_rate * gradients[j]

                self.coef_[-1] -= learning_rate * gradients[-1]

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def compute_gradients(self, x_batch, y_batch):
        gradients = [0.0 for _ in range(len(x_batch[0]) + 1)]

        for i in range(len(x_batch)):
            y_computed = self.eval(x_batch[i])
            crt_error = y_computed - y_batch[i]

            for j in range(len(x_batch[0])):
                gradients[j] += crt_error * x_batch[i][j]
            gradients[-1] += crt_error * 1

        return gradients

    def eval(self, xi):
        yi = self.coef_[-1]

        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]

        return yi

    def predict(self, x):
        y_computed = [self.eval(xi) for xi in x]
        return y_computed


def loadData(fileName, inputVariables, outputVariables):
    data = []
    dataNames = []

    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    selectedVariable = dataNames.index(inputVariables)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]

    selectedOutput = dataNames.index(outputVariables)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def plotDataHistogram(x, variableName):
    plt.hist(x, 10, color='magenta')
    plt.title('histogram of ' + variableName)
    plt.show()


def checkLinearity(inputs, outputs):
    plt.plot(inputs, outputs, 'mo')
    plt.xlabel('capital GDP')
    plt.ylabel('happiness')
    plt.title('capital GDP vs. happiness')
    plt.show()


def splitDataIntoTrainingAndValidationSubsets(inputs, outputs):
    np.random.seed(5)

    indexes = [i for i in range(len(inputs))]

    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if i not in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    return trainInputs, trainOutputs, validationInputs, validationOutputs


def plotTrainAndValidationData(trainInputs, trainOutputs, validationInputs, validationOutputs):
    plt.plot(trainInputs, trainOutputs, 'mo', label='training data')
    plt.plot(validationInputs, validationOutputs, 'y^', label='validation data')
    plt.title('train and validation data')
    plt.xlabel('capital GDP')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def trainingUsingTools(trainInputs, trainOutputs):
    preparedInputs = [[el] for el in trainInputs]
    # regressor = linear_model.LinearRegression()
    # regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=100)
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=1000, learning_rate='constant', eta0=0.01, power_t=0.25)
    regressor.fit(preparedInputs, trainOutputs)
    w0, w1 = regressor.intercept_[0], regressor.coef_[0]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')
    return regressor, w0, w1


def trainingUsingMyCode(trainInputs, trainOutputs):
    preparedInputs = [[el] for el in trainInputs]
    regressor = MyBGDRegression()
    regressor.fit(preparedInputs, trainOutputs)
    w0, w1 = regressor.intercept_, regressor.coef_[0]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')
    return regressor, w0, w1


def plotLearntModel(trainInputs, trainOutputs, w0, w1):
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints

    for i in range(1, noOfPoints):
        xref.append(val)
        val += step

    yref = [w0 + w1 * el for el in xref]

    plt.plot(trainInputs, trainOutputs, 'mo', label='training data')
    plt.plot(xref, yref, 'y-', label='learnt model')
    plt.title('training data and the learnt model')
    plt.xlabel('capital GDP')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def predictNewInputs(regressor, validationInputs):
    computedValidationOutputs = regressor.predict([[x] for x in validationInputs])
    return computedValidationOutputs


def plotPredictedInputs(validationInputs, validationOutputs, computedValidationOutputs):
    plt.plot(validationInputs, computedValidationOutputs, 'yo', label='computed test data')
    plt.plot(validationInputs, validationOutputs, 'm^', label='real test data')
    plt.title('computed test and real test data')
    plt.xlabel('capital GDP')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def checkPerformance(validationOutputs, computedValidationOutputs):
    error = 0.0
    for t1, t2 in zip(computedValidationOutputs, validationOutputs):
        error += (t1 - t2) ** 2
    error = error / len(validationOutputs)
    print('prediction error (manual): ', error)

    error = mean_squared_error(validationOutputs, computedValidationOutputs)
    print('prediction error (tool):  ', error)


def main():
    inputs, outputs = loadData('../data/world-happiness-report-2017.csv', 'Economy..GDP.per.Capita.',
                               'Happiness.Score')
    print('in: ', inputs[:5])
    print('out: ', outputs[:5])
    plotDataHistogram(inputs, 'capital GDP')
    plotDataHistogram(outputs, 'happiness score')

    checkLinearity(inputs, outputs)

    trainInputs, trainOutputs, validationInputs, validationOutputs = splitDataIntoTrainingAndValidationSubsets(inputs,
                                                                                                               outputs)
    plotTrainAndValidationData(trainInputs, trainOutputs, validationInputs, validationOutputs)

    # regressor, w0, w1 = trainingUsingTools(trainInputs, trainOutputs)
    regressor, w0, w1 = trainingUsingMyCode(trainInputs, trainOutputs)
    plotLearntModel(trainInputs, trainOutputs, w0, w1)

    computedValidationOutputs = predictNewInputs(regressor, validationInputs)
    plotPredictedInputs(validationInputs, validationOutputs, computedValidationOutputs)

    checkPerformance(validationOutputs, computedValidationOutputs)


main()
