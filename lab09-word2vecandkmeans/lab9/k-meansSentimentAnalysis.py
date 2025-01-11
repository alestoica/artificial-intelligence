from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy.random import choice
from math import sqrt
import torch
import re
import os
import nltk
import pandas as pd
import numpy as np


# nltk.download('punkt')
# nltk.download('stopwords')


# K-MEANS CLASSIFIER WITHOUT TOOL
class KMeansClassifier:
    def __init__(self, num_centroids) -> None:
        self.num_centroids = num_centroids
        self.centroids = []

    def choose_initial_centroids(self, input_data):
        positions = [i for i in range(input_data.shape[0])]
        centroid_positions = choice(positions, self.num_centroids)
        self.centroids = [input_data[i] for i in centroid_positions]

    def euclidean_distance(self, point1, point2):
        x = [(point1[0, i] - point2[0, i]) ** 2 for i in range(point1.shape[1])]
        distance = sqrt(sum(x))
        return distance

    def closest_centroid_for_point(self, point):
        ind = 0
        min_distance = self.euclidean_distance(point, self.centroids[0])

        for i in range(len(self.centroids)):
            distance = self.euclidean_distance(point, self.centroids[i])
            if distance < min_distance:
                min_distance = distance
                ind = i
        return ind

    def sum_points(self, input_data, clusters, centroid_index):
        return sum([input_data[i] for i in range(input_data.shape[0]) if clusters[i] == centroid_index])

    def count_points(self, clusters, centroid_index):
        return clusters.count(centroid_index)

    def fit(self, training_data):
        self.choose_initial_centroids(training_data)
        convergent = False

        while not convergent:
            clusters = []
            for i in range(training_data.shape[0]):
                point = training_data[i]
                ind = self.closest_centroid_for_point(point)
                clusters.append(ind)

            max_centroid_change = -1
            for centroid_index in range(0, self.num_centroids):
                new_centroid = self.sum_points(training_data, clusters, centroid_index) / self.count_points(clusters,
                                                                                                            centroid_index)
                distance = self.euclidean_distance(self.centroids[centroid_index], new_centroid)

                if distance > max_centroid_change:
                    max_centroid_change = distance
                self.centroids[centroid_index] = new_centroid

            if max_centroid_change < 0.05:
                convergent = True

    def predict(self, input_data):
        return [self.closest_centroid_for_point(i) for i in input_data]


def bagOfWords(trainInput, testInput):
    v = CountVectorizer(max_features=50)

    trainFeatures = v.fit_transform(trainInput)
    testFeatures = v.transform(testInput)

    print()
    print('vocab size: ', len(v.vocabulary_), ' words')
    print('some words of the vocab: ', v.get_feature_names_out()[-10:])
    print('trainInput size: ', len(trainInput), ' words')
    print('trainFeatures shape: ', trainFeatures.shape)
    print('some features: ', trainFeatures.toarray()[:3])
    print()

    return trainFeatures, testFeatures


def tfIdf(trainInput, testInput):
    v = TfidfVectorizer(max_features=50)

    trainFeatures = v.fit_transform(trainInput)
    testFeatures = v.transform(testInput)

    print()
    print('vocab size: ', len(v.vocabulary_), " words")
    print('some words of the vocab: ', v.get_feature_names_out()[-10:])
    print('trainInput size: ', len(trainInput), ' words')
    print('trainFeatures shape: ', trainFeatures.shape)
    print('some features: ', trainFeatures.toarray()[:3])
    print()

    return trainFeatures, testFeatures


def bertEmbeddings(trainInput):
    allEmbeddings = []

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", force_download=True)
    model = BertModel.from_pretrained('bert-base-uncased', force_download=True)

    for input in trainInput:
        input_ids = tokenizer.encode(input, add_special_tokens=True, max_length=128, truncation=True,
                                     return_tensors='pt')

        with torch.no_grad():
            outputs = model(input_ids)
            contextualEmbeddings = outputs.last_hidden_state

        allEmbeddings.append(contextualEmbeddings)

    return allEmbeddings


def stemming(trainInput, testInput):
    trainTokens = [word_tokenize(text) for text in trainInput]
    testTokens = [word_tokenize(text) for text in testInput]

    s = PorterStemmer()
    trainInputStemmedWords = [[s.stem(word) for word in words] for words in trainTokens]
    testInputStemmedWords = [[s.stem(word) for word in words] for words in testTokens]

    trainInputStemmedSentences = [' '.join(words) for words in trainInputStemmedWords]
    testInputStemmedSentences = [' '.join(words) for words in testInputStemmedWords]

    return trainInputStemmedSentences, testInputStemmedSentences


def stopWordsRemoval(trainInput, testInput):
    trainTokens = [word_tokenize(text) for text in trainInput]
    testTokens = [word_tokenize(text) for text in testInput]
    stopWords = set(stopwords.words('english'))

    trainInputStemmedWords = [[word for word in words if word not in stopWords] for words in trainTokens]
    testInputStemmedWords = [[word for word in words if word not in stopWords] for words in testTokens]

    trainInputStemmedSentences = [' '.join(words) for words in trainInputStemmedWords]
    testInputStemmedSentences = [' '.join(words) for words in testInputStemmedWords]

    return trainInputStemmedSentences, testInputStemmedSentences


def specialCharacterRemoval(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def loadData(filePath):
    dataFrame = pd.read_csv(filePath)
    return dataFrame


def getTrainAndTestSubsets(dataFrame):
    np.random.seed(5)
    indexes = [i for i in range(dataFrame.shape[0])]
    trainSample = np.random.choice(indexes, size=int(0.8 * dataFrame.shape[0]), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [dataFrame["Text"].iloc[i] for i in trainSample]
    trainOutputs = [dataFrame["Sentiment"].iloc[i] for i in trainSample]

    testInputs = [dataFrame["Text"].iloc[i] for i in testSample]
    testOutputs = [dataFrame["Sentiment"].iloc[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def getClassifier(classifierType, trainFeatures, numberClusters):
    if classifierType == 'tool':
        classifier = KMeans(n_clusters=numberClusters, random_state=2)

    elif classifierType == 'without tool':
        classifier = KMeansClassifier(numberClusters)

    else:
        classifier = AgglomerativeClustering(n_clusters=numberClusters)

    if classifierType == 'agglomerative':
        classifier.fit(trainFeatures.toarray())
    else:
        classifier.fit(trainFeatures)

    return classifier


def testClassifier(classifierType, classifier, testFeatures, testOutput, labels):
    if classifierType == 'tool' or classifierType == 'without tool':
        computedIndexes = classifier.predict(testFeatures)

    elif classifierType == 'dbscan':
        computedIndexes = classifier.fit_predict(testFeatures)

    else:
        computedIndexes = classifier.fit_predict(testFeatures.toarray())

    computedOutput = [labels[index] for index in computedIndexes]

    print("accuracy: {}\n".format(accuracy_score(testOutput, computedOutput)))


def main(classifierType):
    dataFrame = loadData("data/reviews_mixed.csv")

    dataFrame['Text'] = dataFrame['Text'].apply(specialCharacterRemoval)

    trainInput, trainOutput, testInput, testOutput = getTrainAndTestSubsets(dataFrame)
    print('trainInput: ', trainInput[:5])
    print('testInput: ', testInput[:5])
    print('trainOutput: ', trainOutput[:5])
    print('testOutput: ', testOutput[:5])
    print()

    trainInput, testInput = stemming(trainInput, testInput)
    trainInput, testInput = stopWordsRemoval(trainInput, testInput)
    print('trainInput: ', trainInput[:5])
    print('testInput: ', testInput[:5])
    print('trainOutput: ', trainOutput[:5])
    print('testOutput: ', testOutput[:5])
    print()

    # trainFeatures, testFeatures = bagOfWords(trainInput, testInput)
    trainFeatures, testFeatures = tfIdf(trainInput, testInput)
    # trainFeatures = bertEmbeddings(trainInput)
    # testFeatures = bertEmbeddings(testInput)
    print('trainFeatures:\n', trainFeatures[0])
    print('testFeatures:\n', testFeatures[0])
    print()

    labels = [label for label in set(trainOutput)]
    print('labels: ', labels)
    print()

    classifier = getClassifier(classifierType, trainFeatures, len(labels))
    testClassifier(classifierType, classifier, testFeatures, testOutput, labels)

    # determine the sentiment conveyed by the text
    data = ["By choosing a bike over a car, I’m reducing my environmental footprint. Cycling promotes eco-friendly "
            "transportation, and I’m proud to be part of that movement.."]
    _, dataFeatures = bagOfWords(trainInput, data)
    # _, dataFeatures = tfIdf(trainInput, data)

    # k-means
    computedOutput = classifier.predict(dataFeatures)

    # agglomerative
    # computedOutput = classifier.fit_predict(dataFeatures)

    label = labels[computedOutput[-1]]
    print('the sentiment conveyed by the given text is: ', label)


main('tool')
# main('without tool')
# main('agglomerative')
