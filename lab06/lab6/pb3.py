import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.express as px
import random
import pandas

warnings.simplefilter('ignore')
matplotlib.use('TkAgg')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MyLogisticRegressionMultipleLabels:
    def __init__(self):
        self.intercept_ = []
        self.coef_ = []

    def fit_batch(self, x, y, learning_rate=0.0001, no_epochs=1000):
        self.coef_ = []
        self.intercept_ = []
        labels = list(set(y))

        for label in labels:
            coefficient = [random.random() for _ in range(len(x[0]) + 1)]

            for _ in range(no_epochs):
                errors = [0] * len(coefficient)

                for input, output in zip(x, y):
                    y_computed = sigmoid(self.evaluate(input, coefficient))
                    error = y_computed - 1 if output == label else y_computed

                    for i, xi in enumerate([1] + list(input)):
                        errors[i] += error * xi

                for i in range(len(coefficient)):
                    coefficient[i] = coefficient[i] - learning_rate * errors[i]

            self.intercept_.append(coefficient[0])
            self.coef_.append(coefficient[1:])

    def fit(self, x, y, learning_rate=0.001, no_epochs=1000):
        self.intercept_ = []
        self.coef_ = []
        labels = list(set(y))

        for label in labels:
            coefficient = [random.random() for _ in range(len(x[0]) + 1)]

            for _ in range(no_epochs):
                for input, output in zip(x, y):
                    y_computed = sigmoid(self.evaluate(input, coefficient))
                    error = y_computed - 1 if output == label else y_computed

                    for j in range(len(x[0])):
                        coefficient[j + 1] = coefficient[j + 1] - learning_rate * error * input[j]

                    coefficient[0] = coefficient[0] - learning_rate * error

            self.intercept_.append(coefficient[0])
            self.coef_.append(coefficient[1:])

    def evaluate(self, xi, coefficient):
        yi = coefficient[0]
        for j in range(len(xi)):
            yi += coefficient[j + 1] * xi[j]
        return yi

    def predict_one_sample(self, sample_features):
        predictions = []
        for intercept, coefficient in zip(self.intercept_, self.coef_):
            computed_value = self.evaluate(sample_features, [intercept] + coefficient)
            predictions.append(sigmoid(computed_value))
        return predictions.index(max(predictions))

    def predict(self, in_test):
        computed_labels = [self.predict_one_sample(sample) for sample in in_test]
        return computed_labels


def loadData():
    data = load_iris()

    inputs = data['data']
    outputs = data['target']

    outputs_name = data['target_names']
    feature_names = list(data['feature_names'])

    feature1 = [feat[feature_names.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[feature_names.index('sepal width (cm)')] for feat in inputs]
    feature3 = [feat[feature_names.index('petal length (cm)')] for feat in inputs]
    feature4 = [feat[feature_names.index('petal width (cm)')] for feat in inputs]

    inputs = [[feat[feature_names.index('sepal length (cm)')],
               feat[feature_names.index('sepal width (cm)')],
               feat[feature_names.index('petal length (cm)')],
               feat[feature_names.index('petal width (cm)')]] for feat in inputs]

    return inputs, outputs, feature1, feature2, feature3, feature4, feature_names, outputs_name


def plotDataHistogram(x, variableName):
    plt.hist(x, 10, color='magenta')
    plt.title('histogram of ' + variableName)
    plt.show()


def plotData(inputs, outputs, inputVariables, OutputVariables, title):
    x = [i[0] for i in inputs]
    y = [i[1] for i in inputs]
    z = [i[2] for i in inputs]
    v = [i[3] for i in inputs]

    figure = px.scatter_3d(x=x, y=y, z=z, symbol=v, color=outputs, title=title,
                           labels=dict(x=inputVariables[0], y=inputVariables[1], z=inputVariables[2],
                                       symbol=inputVariables[3], color="Type"))
    figure.update_layout(legend=dict(orientation="v", yanchor='top', xanchor="right"))
    figure.show()


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

    trainInputs, testInputs = normalisationUsingMyCode(trainInputs, testInputs)

    return trainInputs, trainOutputs, testInputs, testOutputs


def trainingUsingTools(trainInputs, trainOutputs):
    classifier = linear_model.LogisticRegression()
    classifier.fit(trainInputs, trainOutputs)

    w0, w1, w2, w3, w4 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[0][1], classifier.coef_[0][
        2], classifier.coef_[0][3]
    print('the learnt model - first label: y =', w0, '+', w1, '* x1 +', w2, '* x2 +', w3, '* x3 +',
          w4, '* x4')

    w0, w1, w2, w3, w4 = classifier.intercept_[1], classifier.coef_[1][0], classifier.coef_[1][1], classifier.coef_[1][
        2], classifier.coef_[1][3]
    print('the learnt model - second label: y =', w0, '+', w1, '* x1 +', w2, '* x2 +', w3, '* x3 +',
          w4, '* x4')

    w0, w1, w2, w3, w4 = classifier.intercept_[2], classifier.coef_[2][0], classifier.coef_[2][1], classifier.coef_[2][
        2], classifier.coef_[2][3]
    print('the learnt model - third label: y =', w0, '+', w1, '* x1 +', w2, '* x2 +', w3, '* x3 +',
          w4, '* x4')

    return classifier, w0, w1, w2, w3, w4


def trainingUsingMyCode(trainInputs, trainOutputs):
    classifier = MyLogisticRegressionMultipleLabels()
    classifier.fit_batch(trainInputs, trainOutputs)

    w0, w1, w2, w3, w4 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[
        0][1], classifier.coef_[0][2], classifier.coef_[0][3]
    print('the learnt model - first label: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')

    w0, w1, w2, w3, w4 = classifier.intercept_[1], classifier.coef_[1][0], classifier.coef_[
        1][1], classifier.coef_[1][2], classifier.coef_[1][3]
    print('the learnt model - second label: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')

    w0, w1, w2, w3, w4 = classifier.intercept_[2], classifier.coef_[2][0], classifier.coef_[
        2][1], classifier.coef_[2][2], classifier.coef_[2][3]
    print('the learnt model - third label: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')

    return classifier, w0, w1, w2, w3, w4


def predictNewInputs(classifier, testInputs):
    computedTestOutputs = classifier.predict(testInputs)
    return computedTestOutputs


def plotPredictedInputs(testInputs, testOutputs, computedTestOutputs, inputVariables, outputVariables):
    labels = set(testOutputs)

    for crt_label in labels:
        xTrain = [testInputs[i][0] for i in range(len(testInputs)) if
                  testOutputs[i] == crt_label and computedTestOutputs[i] == crt_label]
        yTrain = [testInputs[i][1] for i in range(len(testInputs)) if
                  testOutputs[i] == crt_label and computedTestOutputs[i] == crt_label]
        plt.scatter(xTrain, yTrain, label=outputVariables[crt_label] + ' (correct)', marker='o')

        xTest = [testInputs[i][0] for i in range(len(testInputs)) if
                 testOutputs[i] == crt_label and computedTestOutputs[i] != crt_label]
        yTest = [testInputs[i][1] for i in range(len(testInputs)) if
                 testOutputs[i] == crt_label and computedTestOutputs[i] != crt_label]
        plt.scatter(xTest, yTest, label=outputVariables[crt_label] + ' (incorrect)', marker='^')

    plt.xlabel(inputVariables[0])
    plt.ylabel(inputVariables[1])
    plt.legend()
    plt.title('train vs test data')
    plt.show()


def checkPerformance(testOutputs, computedTestOutputs):
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print('prediction error (manual): ', error)

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print('prediction error (tool):   ', error)


def main():
    inputs, outputs, feature1, feature2, feature3, feature4, inputVariables, outputVariables = loadData()

    plotData(inputs, outputs, inputVariables, outputVariables, 'initial data')

    # plotDataHistogram(feature1, inputVariables[0])
    # plotDataHistogram(feature2, inputVariables[1])
    # plotDataHistogram(feature3, inputVariables[2])
    # plotDataHistogram(feature4, inputVariables[3])
    # plotDataHistogram(outputs, 'flower classes')

    trainInputs, trainOutputs, testInputs, testOutputs = splitDataIntoTrainingAndTestSubsets(inputs, outputs)
    plotData(trainInputs, trainOutputs, inputVariables, outputVariables, 'normalised data')

    classifier, w0, w1, w2, w3, w4 = trainingUsingMyCode(trainInputs, trainOutputs)
    # classifier, w0, w1, w2, w3, w4 = trainingUsingTools(trainInputs, trainOutputs)
    # print("Accuracy score:", classifier.score(testInputs, testOutputs))
    computedTestOutputs = predictNewInputs(classifier, testInputs)

    accuracy = 0.0
    for i in range(len(testInputs)):
        if testOutputs[i] == computedTestOutputs[i]:
            accuracy += 1

    print("Accuracy score: ", accuracy / len(testInputs))

    computedTestOutputs = predictNewInputs(classifier, testInputs)
    plotPredictedInputs(testInputs, testOutputs, computedTestOutputs, inputVariables, outputVariables)

    # checkPerformance(testOutputs, computedTestOutputs)


main()
