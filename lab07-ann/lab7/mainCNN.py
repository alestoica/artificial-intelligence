import os
import tensorflow as tf
import matplotlib
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import confusion_matrix
matplotlib.use('TkAgg')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def loadData():
    inputs = []

    for filename in os.listdir('output_images_64'):
        image_path = os.path.join('output_images_64', filename)
        inputs.append(mpimg.imread(image_path))

    inputs = np.array(inputs)
    outputs = np.concatenate([np.zeros(25), np.ones(25)], axis=0)

    classNames = [0, 1]

    permutation = np.random.permutation(len(inputs))
    inputs = inputs[permutation]
    outputs = outputs[permutation]

    inputs = inputs.astype('float32') / 255.0

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

    return np.array(trainInputs), np.array(trainOutputs), np.array(testInputs), np.array(testOutputs)


def evaluateMultiClass(realLabels, computedLabels, classNames):
    confusionMatrix = confusion_matrix(realLabels, computedLabels)

    accuracy = sum([confusionMatrix[i][i] for i in range(len(classNames))]) / len(realLabels)
    precision = {}
    recall = {}

    for i in range(len(classNames)):
        precision[classNames[i]] = confusionMatrix[i][i] / sum([confusionMatrix[j][i] for j in range(len(classNames))])
        recall[classNames[i]] = confusionMatrix[i][i] / sum([confusionMatrix[i][j] for j in range(len(classNames))])

    return accuracy, precision, recall, confusionMatrix


def main():
    inputs, outputs, classNames = loadData()

    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

    model = CNN()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(trainInputs, trainOutputs, epochs=10, batch_size=32, validation_data=(testInputs, testOutputs))

    predictions = model.predict(testInputs)
    predictedLabels = np.argmax(predictions, axis=1)

    testOutputs = testOutputs.astype(int)

    accuracy, precision, recall, confusionMatrix = evaluateMultiClass(np.array(testOutputs), predictedLabels,
                                                                      classNames)
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)


main()

