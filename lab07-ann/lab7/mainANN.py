import os
import matplotlib
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import itertools

matplotlib.use('TkAgg')


class MyANN:
    def __init__(self, hidden_layer_sizes=(5,), max_iter=100, learning_rate_init=0.1, random_state=None, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.verbose = verbose

        self.coefs_ = []
        self.intercepts_ = []
        self.activation_function = self.sigmoid
        self.activation_derivative = self.sigmoid_derivative

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        # Initialize weights and biases
        np.random.seed(self.random_state)
        layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes) + [len(np.unique(y))]

        for i in range(len(layer_sizes) - 1):
            # Initialize weights and biases for each layer
            self.coefs_.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.intercepts_.append(np.random.randn(layer_sizes[i + 1]))

        # Stochastic Gradient Descent
        for epoch in range(self.max_iter):
            loss = 0

            for i in range(X.shape[0]):
                # Forward pass
                activations = [X[i]]
                for j in range(len(self.coefs_)):
                    net_input = np.dot(activations[j], self.coefs_[j]) + self.intercepts_[j]
                    activation = self.activation_function(net_input)
                    activations.append(activation)

                # Calculate the loss
                loss += self.mean_squared_error(y[i], activations[-1])

                # Backward pass
                error = y[i] - activations[-1]
                deltas = [error * self.activation_derivative(activations[-1])]

                for j in range(len(activations) - 2, 0, -1):
                    delta = np.dot(deltas[-1], self.coefs_[j].T) * self.activation_derivative(activations[j])
                    deltas.append(delta)

                deltas.reverse()

                # Update weights and biases
                for j in range(len(self.coefs_)):
                    self.coefs_[j] += self.learning_rate_init * np.outer(activations[j], deltas[j])
                    self.intercepts_[j] += self.learning_rate_init * deltas[j]

            avg_loss = loss / X.shape[0]
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_iter} - Loss: {avg_loss:.4f}")

    def predict(self, X):
        activations = [X]

        for j in range(len(self.coefs_)):
            net_input = np.dot(activations[j], self.coefs_[j]) + self.intercepts_[j]
            activation = self.activation_function(net_input)
            activations.append(activation)

        # Convert to class labels
        predictions = np.argmax(activations[-1], axis=1)
        return predictions

    def predict_proba(self, X):
        activations = [X]

        for j in range(len(self.coefs_)):
            net_input = np.dot(activations[j], self.coefs_[j]) + self.intercepts_[j]
            activation = self.activation_function(net_input)
            activations.append(activation)

        # Convert to class probabilities
        probabilities = activations[-1]
        return probabilities


def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x


def loadData():
    data = load_digits()

    inputs = data.images

    outputs = data['target']

    classNames = data['target_names']

    inputs = [flatten(inp) for inp in inputs]

    noData = len(inputs)
    permutation = np.random.permutation(noData)

    inputs = [inputs[perm] for perm in permutation]

    outputs = [outputs[perm] for perm in permutation]

    return inputs, outputs, classNames


def loadDataNew():
    inputs = []

    for filename in os.listdir('output_images_64'):
        image_path = os.path.join('output_images_64', filename)
        inputs.append(mpimg.imread(image_path))

    outputs = []

    for _ in range(25):
        outputs.append(0)

    for _ in range(25):
        outputs.append(1)

    classNames = [0, 1]

    inputs = [flatten(inp) for inp in inputs]
    inputs = [flatten(inp) for inp in inputs]

    noData = len(inputs)
    permutation = np.random.permutation(noData)

    inputs = [inputs[perm] for perm in permutation]

    outputs = [outputs[perm] for perm in permutation]

    return inputs, outputs, classNames


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def normalisation(trainData, testData):
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
    return normalisedTrainData, normalisedTestData


def plotDataHistogram(inputs, classNames):
    bins = range(len(classNames) + 1)
    plt.hist(inputs, bins, rwidth=0.8)
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w), classNames)
    plt.title('training output data distribution on classes (histogram)')
    plt.show()


def plotData2FeaturesMoreClasses(inputs, outputs, classNames):
    labels = set(outputs)

    for crtLabel in labels:
        x = [inputs[i][0] for i in range(len(inputs)) if outputs[i] == crtLabel]
        y = [inputs[i][1] for i in range(len(inputs)) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=classNames[crtLabel])

    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend()
    plt.title('plot data')
    plt.show()


def trainUsingTool(trainInputs, trainOutputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(3,), max_iter=100, solver='sgd',
                                              verbose=10, random_state=1, learning_rate_init=0.001)
    classifier.fit(trainInputs, trainOutputs)
    return classifier


def trainUsingMyANN(trainInputs, trainOutputs):
    classifier = MyANN(hidden_layer_sizes=(3, 3), max_iter=300, verbose=True, random_state=1, learning_rate_init=0.01)

    classifier.fit(trainInputs, trainOutputs)
    return classifier


def testModel(classifier, testInputs):
    computedLabels = classifier.predict(testInputs)
    return computedLabels


def evaluateMultiClass(realLabels, computedLabels, classNames):
    confusionMatrix = confusion_matrix(realLabels, computedLabels)
    accuracy = sum([confusionMatrix[i][i] for i in range(len(classNames))]) / len(realLabels)
    return accuracy, confusionMatrix


def plotConfusionMatrix(confusionMatrix, classNames, title):
    plt.figure()
    plt.imshow(confusionMatrix, interpolation='nearest', cmap='Oranges')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    text_format = 'd'
    thresh = confusionMatrix.max() / 2.
    for row, column in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(column, row, format(confusionMatrix[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if confusionMatrix[row, column] > thresh else 'black')

    plt.ylabel('Real label')
    plt.xlabel('Computed label')
    plt.tight_layout()

    plt.show()


def main():
    # step1: load the data
    inputs, outputs, classNames = loadDataNew()
    print("class names: ", classNames)

    # step2: split data into train and test
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

    # plot the training data distribution on classes
    plotDataHistogram(trainOutputs, classNames)

    # normalise the data
    trainInputs, testInputs = normalisation(trainInputs, testInputs)

    # step3: training the classifier
    # # linear classifier and one-vs-all approach for multi-class
    # # classifier = linear_model.LogisticRegression()
    # # non-linear classifier and softmax approach for multi-class
    # # classifier = neural_network.MLPClassifier()
    classifier = trainUsingTool(trainInputs, trainOutputs)
    classifier = trainUsingMyANN(trainInputs, trainOutputs)

    # step 4: test the model (predict new outputs for the testInputs)
    computedLabels = testModel(classifier, testInputs)

    # step 5: calculate the performance metrics
    accuracy, confusionMatrix = evaluateMultiClass(np.array(testOutputs), computedLabels,
                                                   classNames)
    print('accuracy: ', accuracy)
    plotConfusionMatrix(confusionMatrix, classNames, "normal vs. sepia classification")


main()
