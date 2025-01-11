import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import random
from statistics import mean

warnings.simplefilter('ignore')
matplotlib.use('TkAgg')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MyLogisticRegression:
    def __init__(self, threshold):
        self.threshold = threshold
        self.intercept_ = 0.0
        self.coef_ = []
        self.loss = []

    def fit(self, x, y, learning_rate=0.001, no_epochs=1000):
        self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]

        for epoch in range(no_epochs):
            epoch_loss = []

            for i in range(len(x)):
                y_computed = sigmoid(self.evaluate(x[i], self.coef_))
                # print(y_computed)
                crt_error = y_computed - y[i]
                epoch_loss.append(crt_error)

                for j in range(0, len(x[0])):
                    self.coef_[j + 1] = self.coef_[j + 1] - learning_rate * crt_error * x[i][j]

                self.coef_[0] = self.coef_[0] - learning_rate * crt_error * 1

            self.loss.append(mean(epoch_loss))

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def evaluate(self, xi, coefficient):
        yi = coefficient[0]
        for j in range(len(xi)):
            yi += coefficient[j + 1] * xi[j]
        return yi

    def predict_one_sample(self, sample_features):
        threshold = self.threshold
        coefficients = [self.intercept_] + [c for c in self.coef_]
        computed_float_value = self.evaluate(sample_features, coefficients)
        computed01_value = sigmoid(computed_float_value)
        computed_label = 0 if computed01_value < threshold else 1
        return computed_label

    def predict(self, in_test):
        computed_labels = [self.predict_one_sample(sample) for sample in in_test]
        return computed_labels


def loadData():
    data = load_breast_cancer()

    inputs = data['data']
    outputs = data['target']

    feature1 = [feat[0] for feat in inputs]
    feature2 = [feat[1] for feat in inputs]

    inputs = [[feat[0], feat[1]] for feat in
              inputs]

    return inputs, outputs, feature1, feature2


def plotData(inputs, outputs, inputVariables, outputVariables, title):
    labels = set(outputs)

    for crt_label in labels:
        x = [inputs[i][0] for i in range(len(inputs)) if outputs[i] == crt_label]
        y = [inputs[i][1] for i in range(len(inputs)) if outputs[i] == crt_label]
        plt.scatter(x, y, label=outputVariables[crt_label])

    plt.xlabel(inputVariables[0])
    plt.ylabel(inputVariables[1])
    plt.legend()
    plt.title(title)
    plt.show()


def plotDataHistogram(x, variableName):
    plt.hist(x, 10, color='magenta')
    plt.title('histogram of ' + variableName)
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

    trainInputs, testInputs = normalisationUsingMyCode(trainInputs, testInputs)

    return trainInputs, trainOutputs, testInputs, testOutputs


def plotTrainAndValidationData(trainInputs, trainOutputs, testInputs, testOutputs, inputVariables, outputVariables):
    labels = set(trainOutputs)

    for crt_label in labels:
        xTrain = [trainInputs[i][0] for i in range(len(trainOutputs)) if trainOutputs[i] == crt_label]
        yTrain = [trainInputs[i][1] for i in range(len(trainInputs)) if trainOutputs[i] == crt_label]
        plt.scatter(xTrain, yTrain, label='train ' + outputVariables[crt_label], marker='o')

        xTest = [testInputs[i][0] for i in range(len(testInputs)) if testOutputs[i] == crt_label]
        yTest = [testInputs[i][1] for i in range(len(testInputs)) if testOutputs[i] == crt_label]
        plt.scatter(xTest, yTest, label='test ' + outputVariables[crt_label], marker='^')

    plt.xlabel(inputVariables[0])
    plt.ylabel(inputVariables[1])
    plt.legend()
    plt.title('train vs test data')
    plt.show()


def trainingUsingTools(trainInputs, trainOutputs):
    classifier = linear_model.LogisticRegression()
    classifier.fit(trainInputs, trainOutputs)
    w0, w1, w2 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[0][1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, ' * x2')
    return classifier, w0, w1, w2


def trainingUsingMyCode(trainInputs, trainOutputs):
    classifier = MyLogisticRegression(0.5)
    classifier.fit(trainInputs, trainOutputs)
    w0, w1, w2 = classifier.intercept_, classifier.coef_[0], classifier.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, ' * x2')
    return classifier, w0, w1, w2


def predictNewInputs(regressor, testInputs):
    computedTestOutputs = regressor.predict(testInputs)
    return computedTestOutputs


def plotPredictedInputs(testInputs, testOutputs, computedTestOutputs, inputVariables, outputVariables, title):
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
    plt.title(title)
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
    ax.set_xlabel("mean radius")
    ax.set_ylabel("mean texture")
    ax.set_zlabel("cancer")

    plt.legend()
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
    inputVariables = ['mean radius', 'mean texture']
    outputVariables = ['malignant', 'benign']

    inputs, outputs, feature1, feature2 = loadData()
    plotData(inputs, outputs, inputVariables, outputVariables, 'initial data')

    # plotDataHistogram(feature1, 'mean radius')
    # plotDataHistogram(feature2, 'mean texture')
    # plotDataHistogram(outputs, 'outputs')

    trainInputs, trainOutputs, testInputs, testOutputs = splitDataIntoTrainingAndTestSubsets(inputs, outputs)
    plotTrainAndValidationData(trainInputs, trainOutputs, testInputs, testOutputs, inputVariables, outputVariables)

    # regressor, w0, w1, w2 = trainingUsingTools(trainInputs, trainOutputs)
    # print("Accuracy score:", regressor.score(trainInputs, trainOutputs))
    regressor, w0, w1, w2 = trainingUsingMyCode(trainInputs, trainOutputs)
    computedTestOutputs = predictNewInputs(regressor, testInputs)
    accuracy = 0.0
    for i in range(len(testInputs)):
        if testOutputs[i] == computedTestOutputs[i]:
            accuracy += 1
    print("Accuracy score:", accuracy / len(testInputs))
    # plotPredictedInputs(testInputs, testOutputs, computedTestOutputs, inputVariables, outputVariables, 'train vs test data')
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]
    plotData3D([], [], [], feature1test, feature2test, computedTestOutputs, feature1test, feature2test, testOutputs,
               'predictions vs real test data')

    # checkPerformance(testOutputs, computedTestOutputs)


def crossValidation(inputs, outputs):
    indexes = [i for i in range(len(inputs))]
    first_set_index = []
    second_set_index = []
    third_set_index = []
    forth_set_index = []
    fifth_set_index = []

    for i in range(5):
        first_set_index = np.random.choice(indexes, int(0.2 * len(inputs)), replace=False)
        used = list(first_set_index)
        second_set_index = np.random.choice([i for i in indexes if i not in used], int(0.2 * len(inputs)),
                                            replace=False)
        used += list(second_set_index)
        third_set_index = np.random.choice([i for i in indexes if i not in used], int(0.20 * len(inputs)),
                                           replace=False)
        used += list(third_set_index)
        forth_set_index = np.random.choice([i for i in indexes if i not in used], int(0.20 * len(inputs)),
                                           replace=False)
        used += list(forth_set_index)
        fifth_set_index = [i for i in indexes if i not in used]

    first_set = {
        'inputs': [inputs[i] for i in first_set_index],
        'outputs': [outputs[i] for i in first_set_index]
    }
    second_set = {
        'inputs': [inputs[i] for i in second_set_index],
        'outputs': [outputs[i] for i in second_set_index]
    }
    third_set = {
        'inputs': [inputs[i] for i in third_set_index],
        'outputs': [outputs[i] for i in third_set_index]
    }
    forth_set = {
        'inputs': [inputs[i] for i in forth_set_index],
        'outputs': [outputs[i] for i in forth_set_index]
    }
    fifth_set = {
        'inputs': [inputs[i] for i in fifth_set_index],
        'outputs': [outputs[i] for i in fifth_set_index]
    }

    return [first_set, second_set, third_set, forth_set, fifth_set]


def other_loss_function(train_inputs, train_outputs, test_inputs, test_outputs):
    classifier = linear_model.SGDClassifier(loss='log_loss')
    classifier.fit(train_inputs, train_outputs)
    print('Accuracy score (log loss by tool):', classifier.score(test_inputs, test_outputs))

    classifier = linear_model.SGDClassifier(loss='hinge')
    classifier.fit(train_inputs, train_outputs)
    print('Accuracy score (hinge loss by tool):', classifier.score(test_inputs, test_outputs))

    classifier = linear_model.SGDClassifier(loss='squared_hinge')
    classifier.fit(train_inputs, train_outputs)
    print('Accuracy score (squared hinge loss by tool):', classifier.score(test_inputs, test_outputs))


def mainOptional():
    inputVariables = ['mean radius', 'mean texture']
    outputVariables = ['malignant', 'benign']
    inputs, outputs, feature1, feature2 = loadData()

    print("Cross validation:")

    data = crossValidation(inputs, outputs)

    for index in range(5):
        testInputs = data[index]['inputs']
        testOutputs = data[index]['outputs']
        trainInputs = []
        trainOutputs = []

        for dictionary in data[:index] + data[index + 1:]:
            trainInputs += dictionary['inputs']
            trainOutputs += dictionary['outputs']

        trainInputs, testInputs = normalisationUsingMyCode(trainInputs, testInputs)

        regressor, w0, w1, w2 = trainingUsingTools(trainInputs, trainOutputs)
        computedTestOutputs = predictNewInputs(regressor, testInputs)

        checkPerformance(testOutputs, computedTestOutputs)

        print()

    print("Other loss functions:")
    trainInputs, trainOutputs, testInputs, testOutputs = splitDataIntoTrainingAndTestSubsets(inputs, outputs)
    other_loss_function(trainInputs, trainOutputs, testInputs, testOutputs)

    print()

    print("Bigger threshold:")
    # classifier = MyLogisticRegression(0.8)
    classifier = linear_model.LogisticRegression()
    classifier.fit(trainInputs, trainOutputs)
    w0, w1, w2 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[0][1]

    computedTestOutputs = [1 if sigmoid(w0 + w1 * el[0] +
                                        w2 * el[1]) > 0.8 else 0 for el in testInputs]

    accuracy = 0.0
    for i in range(len(testInputs)):
        if testOutputs[i] == computedTestOutputs[i]:
            accuracy += 1

    # print("Accuracy score: ", accuracy / len(testInputs))
    print('Accuracy score:', classifier.score(testInputs, testOutputs))

    plotPredictedInputs(testInputs, testOutputs, computedTestOutputs, inputVariables, outputVariables,
                        'Results with threshold 0.8')

    print()

    print("Smaller threshold:")
    # classifier = MyLogisticRegression(0.3)
    classifier.fit(trainInputs, trainOutputs)
    w0, w1, w2 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[0][1]

    computedTestOutputs = [1 if sigmoid(w0 + w1 * el[0] +
                                        w2 * el[1]) > 0.3 else 0 for el in testInputs]

    accuracy = 0.0
    for i in range(len(testInputs)):
        if testOutputs[i] == computedTestOutputs[i]:
            accuracy += 1

    # print("Accuracy score: ", accuracy / len(testInputs))
    print('Accuracy score:', classifier.score(testInputs, testOutputs))

    plotPredictedInputs(testInputs, testOutputs, computedTestOutputs, inputVariables, outputVariables,
                        'Results with threshold 0.3')


# main()
mainOptional()
