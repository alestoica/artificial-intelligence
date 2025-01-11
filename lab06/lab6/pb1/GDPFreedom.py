import warnings
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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

    selectedVariables = [dataNames.index(var) for var in inputVariables]
    inputs = [[float(data[i][selectedVariable]) for selectedVariable in selectedVariables] for i in range(len(data))]

    selectedOutput = dataNames.index(outputVariables)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def plotDataHistogram(x, variableName):
    plt.hist(x, 10, color='magenta')
    plt.title('histogram of ' + variableName)
    plt.show()


def plotData3D(x1Train, x2Train, yTrain, x1Model, x2Model, yModel, x1Test, x2Test,
               yTest, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if x1Train:
        ax.scatter(x1Train, x2Train, yTrain, color='m', marker='o', label='train data')

    if x1Model:
        plt.scatter(x1Model, x2Model, yModel, color='y', marker='_', label='learnt model')

    if x1Test:
        ax.scatter(x1Test, x2Test, yTest, color='g', marker='^', label='test data')

    plt.title(title)
    ax.set_xlabel("capital GDP")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")

    plt.legend()
    plt.show()


def checkLinearity(inputs, outputs):
    plt.scatter([input[0] for input in inputs], [input[1] for input in inputs], c=outputs, cmap='plasma')
    plt.xlabel('GDP capital')
    plt.ylabel('freedom')
    plt.title('GDP capital vs. freedom')
    plt.colorbar(label='happiness score')
    plt.show()


def normalisationUsingTools(trainData, testData):
    scaler = StandardScaler()

    if not isinstance(trainData[0], list):
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)

        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)

    return list(normalisedTrainData), list(normalisedTestData)


def normalisationUsingMyCode(trainData, testData):
    means = np.mean(trainData, axis=0)
    std_devs = np.std(trainData, axis=0)

    if not isinstance(trainData[0], list):
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        normalisedTrainData = (trainData - means) / std_devs
        normalisedTestData = (testData - means) / std_devs

        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        normalisedTrainData = (trainData - means) / std_devs
        normalisedTestData = (testData - means) / std_devs

    return list(normalisedTrainData), list(normalisedTestData)


def splitDataIntoTrainingAndTestSubsets(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # trainInputs, testInputs = normalisationUsingTools(trainInputs, testInputs)
    # trainOutputs, testOutputs = normalisationUsingTools(trainOutputs, testOutputs)

    trainInputs, testInputs = normalisationUsingMyCode(trainInputs, testInputs)
    trainOutputs, testOutputs = normalisationUsingMyCode(trainOutputs, testOutputs)

    return trainInputs, trainOutputs, testInputs, testOutputs


def plotTrainingAndTestData(trainInputs, trainOutputs, testInputs, testOutputs):
    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]
    plotData3D(feature1train, feature2train, trainOutputs, [], [], [], feature1test, feature2test, testOutputs,
               "train and test data (after normalisation)")


def trainingUsingTools(trainInputs, trainOutputs):
    # regressor = linear_model.LinearRegression()
    # regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=100)
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=1000, learning_rate='constant', eta0=0.01, power_t=0.25)
    regressor.fit(trainInputs, trainOutputs)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, ' * x2')
    return regressor, w0, w1, w2


def trainingUsingMyCode(trainInputs, trainOutputs):
    regressor = MyBGDRegression()
    regressor.fit(trainInputs, trainOutputs)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, ' * x2')
    return regressor, w0, w1, w2


def plotLearntModel(feature1, feature2, w0, w1, w2, trainInputs, trainOutputs):
    noOfPoints = 50
    xref1 = []
    val = min(feature1)
    step1 = (max(feature1) - min(feature1)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1

    xref2 = []
    val = min(feature2)
    step2 = (max(feature2) - min(feature2)) / noOfPoints
    for _ in range(1, noOfPoints):
        aux = val
        for _ in range(1, noOfPoints):
            xref2.append(aux)
            aux += step2

    yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]

    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]

    plotData3D(feature1train, feature2train, trainOutputs, xref1, xref2, yref, [], [], [],
               'train data and the learnt model')


def predictNewInputs(regressor, testInputs):
    computedTestOutputs = regressor.predict(testInputs)
    return computedTestOutputs


def plotPredictedInputs(testInputs, testOutputs, computedTestOutputs):
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]
    plotData3D([], [], [], feature1test, feature2test, computedTestOutputs, feature1test, feature2test, testOutputs,
               'predictions vs real test data')


def checkPerformance(testOutputs, computedTestOutputs):
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print('prediction error (manual): ', error)

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print('prediction error (tool):   ', error)


def main():
    inputVariables = ['Economy..GDP.per.Capita.', 'Freedom']
    outputVariable = 'Happiness.Score'

    inputs, outputs = loadData('../data/world-happiness-report-2017.csv', inputVariables,
                               outputVariable)

    print('in: ', inputs[:5])
    print('out: ', outputs[:5])

    feature1 = [ex[0] for ex in inputs]
    feature2 = [ex[1] for ex in inputs]

    # plotDataHistogram(feature1, 'capital GDP')
    # plotDataHistogram(feature2, 'freedom')
    # plotDataHistogram(outputs, 'happiness score')

    plotData3D(feature1, feature2, outputs, [], [], [], [], [], [], 'capital GDP vs freedom vs happiness')

    # checkLinearity(inputs, outputs)

    trainInputs, trainOutputs, testInputs, testOutputs = splitDataIntoTrainingAndTestSubsets(inputs, outputs)
    # plotTrainingAndTestData(trainInputs, trainOutputs, testInputs, testOutputs)

    regressor, w0, w1, w2 = trainingUsingTools(trainInputs, trainOutputs)
    # regressor, w0, w1, w2 = trainingUsingMyCode(trainInputs, trainOutputs)
    plotLearntModel(feature1, feature2, w0, w1, w2, trainInputs, trainOutputs)

    computedTestOutputs = predictNewInputs(regressor, testInputs)
    plotPredictedInputs(testInputs, testOutputs, computedTestOutputs)

    # checkPerformance(testOutputs, computedTestOutputs)


main()
