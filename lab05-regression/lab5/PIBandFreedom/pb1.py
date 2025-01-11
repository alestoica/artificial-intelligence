import matplotlib
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')


def loadData(fileName, inputVariables, outputVariableName):
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

    selectedVariables = [dataNames.index(var) for var in inputVariables]
    inputs = [[float(data[i][idx]) for idx in selectedVariables] for i in range(len(data))]

    selectedOutput = dataNames.index(outputVariableName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def plotDataHistogram(x, variableName):
    plt.hist(x, 10, color='magenta')
    plt.title('histogram of ' + variableName)
    plt.show()


def checkLinearity(inputs, outputs):
    plt.scatter([input[0] for input in inputs], [input[1] for input in inputs], c=outputs, cmap='plasma')
    plt.xlabel('GDP capital')
    plt.ylabel('freedom')
    plt.title('GDP capital vs. freedom')
    plt.colorbar(label='happiness score')
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
    plt.scatter([input[0] for input in trainInputs], [input[1] for input in trainInputs], c=trainOutputs,
                cmap='plasma', label='training data')

    plt.scatter([input[0] for input in validationInputs], [input[1] for input in validationInputs], c=validationOutputs,
                cmap='plasma', marker='^', label='validation data')

    plt.xlabel('GDP capital')
    plt.ylabel('freedom')
    plt.title('training and validation data')
    plt.colorbar(label='happiness score')
    plt.legend()
    plt.show()


def trainingUsingTools(trainInputs, trainOutputs):
    # model initialisation
    regressor = linear_model.LinearRegression()

    # training the model by using the training inputs and known training outputs
    regressor.fit(trainInputs, trainOutputs)

    # save the model parameters
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * GDP capital + ', w2, ' * Freedom')
    return regressor, w0, w1, w2


def visualizeLearntModel(trainInputs, trainOutputs, w0, w1, w2):
    x1 = np.linspace(min([input[0] for input in trainInputs]), max([input[0] for input in trainInputs]), 1000)
    x2 = np.linspace(min([input[1] for input in trainInputs]), max([input[1] for input in trainInputs]), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Y = w0 + w1 * X1 + w2 * X2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([input[0] for input in trainInputs], [input[1] for input in trainInputs], trainOutputs, color='magenta',
               marker='o', label='training data')
    ax.plot_surface(X1, X2, Y, color='blue', alpha=0.5, label='learnt data')
    ax.set_xlabel('GDP capital')
    ax.set_ylabel('freedom')
    ax.set_zlabel('happiness score')
    ax.set_title('training data and the learnt model')
    plt.legend()
    plt.show()


def predictNewInputs(regressor, validationInputs, validationOutputs):
    computedValidationOutputs = regressor.predict(validationInputs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([input[0] for input in validationInputs], [input[1] for input in validationInputs],
               computedValidationOutputs, color='magenta', marker='o', label='computed validation data')
    ax.scatter([input[0] for input in validationInputs], [input[1] for input in validationInputs], validationOutputs,
               color='blue', marker='^', label='real validation data')
    ax.set_xlabel('GDP capital')
    ax.set_ylabel('freedom')
    ax.set_zlabel('happiness score')
    ax.set_title('computed validation and real validation data')
    plt.legend()
    plt.show()
    return computedValidationOutputs


def checkPerformance(validationOutputs, computedValidationOutputs):
    error = mean_squared_error(validationOutputs, computedValidationOutputs)
    print('prediction error (tool):  ', error)


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, '../data', 'v1_world-happiness-report-2017.csv')

    inputVariables = ['Economy..GDP.per.Capita.', 'Freedom']
    outputVariableName = 'Happiness.Score'
    inputs, outputs = loadData(filePath, inputVariables, outputVariableName)
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])

    plotDataHistogram([input[0] for input in inputs], 'capital GDP')
    plotDataHistogram([input[1] for input in inputs], 'freedom')

    checkLinearity(inputs, outputs)

    trainInputs, trainOutputs, validationInputs, validationOutputs = \
        splitDataIntoTrainingAndValidationSubsets(inputs, outputs)
    visualizeTrainAndValidationData(trainInputs, trainOutputs, validationInputs, validationOutputs)

    regressor, w0, w1, w2 = trainingUsingTools(trainInputs, trainOutputs)
    visualizeLearntModel(trainInputs, trainOutputs, w0, w1, w2)

    computedValidationOutputs = predictNewInputs(regressor, validationInputs, validationOutputs)

    checkPerformance(validationOutputs, computedValidationOutputs)


main()
