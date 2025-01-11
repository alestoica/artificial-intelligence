import matplotlib
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')


def loadData(fileName, inputVariableName, outputVariableName):
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

    selectedVariable = dataNames.index(inputVariableName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]

    selectedOutput = dataNames.index(outputVariableName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def plotDataHistogram(x, variableName):
    plt.hist(x, 10)
    plt.title('histogram of ' + variableName)
    plt.show()


def checkLinearity(inputs, outputs):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel('family')
    plt.ylabel('happiness')
    plt.title('family vs. happiness')
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


def visualizeTrainAndValidationData(trainInputs, trainOutputs, validationInputs, validationOutputs):
    plt.plot(trainInputs, trainOutputs, 'ro', label='training data')
    plt.plot(validationInputs, validationOutputs, 'g^', label='validation data')
    plt.title('train and validation data')
    plt.xlabel('family')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def trainingUsingTools(trainInputs, trainOutputs):
    # training data preparation
    # the sklearn linear model requires as input training data as noSamples x noFeatures array
    # in the current case, the input must be a matrix of len(trainInputs) lines
    # and one columns (a single feature is used in this problem)
    preparedInputs = [[el] for el in trainInputs]

    # model initialisation
    regressor = linear_model.LinearRegression()

    # training the model by using the training inputs and known training outputs
    regressor.fit(preparedInputs, trainOutputs)

    # save the model parameters
    w0, w1 = regressor.intercept_, regressor.coef_[0]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * GDP capital')
    return regressor, w0, w1


def trainingManually(trainInputs, trainOutputs):
    from myRegression import MyLinearUnivariateRegression
    regressor = MyLinearUnivariateRegression()
    regressor.fit(trainInputs, trainOutputs)
    w0, w1 = regressor.intercept_, regressor.coef_
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * Family')
    return w0, w1


def visualizeLearntModel(trainInputs, trainOutputs, w0, w1):
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints

    for i in range(1, noOfPoints):
        xref.append(val)
        val += step

    yref = [w0 + w1 * el for el in xref]

    plt.plot(trainInputs, trainOutputs, 'ro', label='training data')
    plt.plot(xref, yref, 'b-', label='learnt model')
    plt.title('training data and the learnt model')
    plt.xlabel('family')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def predictNewInputs(regressor, validationInputs, validationOutputs):
    computedValidationOutputs = regressor.predict([[x] for x in validationInputs])

    plt.plot(validationInputs, computedValidationOutputs, 'yo', label='computed validation data')
    plt.plot(validationInputs, validationOutputs, 'g^', label='real validation data')
    plt.title('computed validation and real validation data')
    plt.xlabel('family')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()

    return computedValidationOutputs


def checkPerformance(validationOutputs, computedValidationOutputs):
    error = 0.0
    for t1, t2 in zip(computedValidationOutputs, validationOutputs):
        error += (t1 - t2) ** 2
    error = error / len(validationOutputs)
    print('prediction error (manual): ', error)

    error = mean_squared_error(validationOutputs, computedValidationOutputs)
    print('prediction error (tool):  ', error)


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, '../data', 'v1_world-happiness-report-2017.csv')

    inputs, outputs = loadData(filePath, 'Family', 'Happiness.Score')
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])

    plotDataHistogram(inputs, 'family')
    plotDataHistogram(outputs, 'happiness score')

    checkLinearity(inputs, outputs)

    trainInputs, trainOutputs, validationInputs, validationOutputs = \
        splitDataIntoTrainingAndValidationSubsets(inputs, outputs)
    visualizeTrainAndValidationData(trainInputs, trainOutputs, validationInputs, validationOutputs)

    regressor, w0, w1 = trainingUsingTools(trainInputs, trainOutputs)
    visualizeLearntModel(trainInputs, trainOutputs, w0, w1)

    computedValidationOutputs = predictNewInputs(regressor, validationInputs, validationOutputs)
    checkPerformance(validationOutputs, computedValidationOutputs)


main()
