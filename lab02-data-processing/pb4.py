import csv
import math
import os

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from nltk import sent_tokenize, word_tokenize
from nltk.internals import Counter

matplotlib.use('TkAgg')


def loadDataMoreInputs(fileName):
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

    return dataNames, data


def extractFeature(allData, names, featureName):
    pos = names.index(featureName)
    return [float(data[pos]) for data in allData]


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 20)
    plt.title('Histogram of ' + variableName)
    plt.xticks(rotation=90)
    plt.show()


def pb1(feature):
    """
    Visualizes the distribution of a feature in a dataset.

    Specifications:
    1. Reads a dataset from a CSV file named 'employees.csv' in the 'data' directory.
    2. If the feature is 'Team':
       - Drops rows with missing values in the 'Team' column.
       - Maps each unique team to a numerical value and visualizes the distribution.
    3. If the feature is not 'Team':
       - Loads the dataset using custom functions loadDataMoreInputs and extractFeature.
       - Scales the feature values to the range [0, 1] and visualizes the distribution.
    4. Displays histograms of the feature and scaled feature side by side.

    Dependencies:
    - matplotlib.pyplot: for data visualization.
    - pandas: for data manipulation.

    Parameters:
    feature (str): The feature to visualize.

    Returns:
    None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if feature == 'Team':
        df_employees = pd.read_csv('data/employees.csv')
        df_employees = df_employees.dropna(subset=[feature])

        team_mapping = {team: i for i, team in enumerate(df_employees[feature].unique())}

        x = df_employees[feature].astype(str)

        scaledX = df_employees[feature].map(team_mapping).astype(str)
        scaledX = scaledX.map({str(v): str(k) for k, v in team_mapping.items()})
        print(scaledX)

        ax1.set_xticklabels(x, rotation=90)
        ax2.set_xticklabels(x, rotation=90)

        ax1.hist(x, 20)
        ax1.set_title(feature + ' histo')

        ax2.hist(scaledX, 20)
        ax2.set_title('Scaled ' + feature + ' histo')

        plt.show()
    else:
        names, data = loadDataMoreInputs('data/employees.csv')
        x = extractFeature(data, names, feature)
        scaledX = [(s - min(x)) / (max(x) - min(x)) for s in x]

        ax1.hist(x, 20)
        ax1.set_title(feature + ' histo')

        ax2.hist(scaledX, 20)
        ax2.set_title('[0,1] scaled ' + feature + ' histo')

        plt.show()


# def pb1(feature):
#     df_employees = pd.read_csv('data/employees.csv', delimiter=',', header='infer')
#
#     if feature == 'Team':
#         team_mapping = {team: i for i, team in enumerate(df_employees['Team'].unique())}
#         x = df_employees['Team'].astype(str)
#         scaledX = df_employees['Team'].map(team_mapping).astype(str)
#         scaledX = scaledX.map({str(v): str(k) for k, v in team_mapping.items()})
#     else:
#         x = df_employees[feature]
#         scaledX = (x - x.min()) / (x.max() - x.min())  # Min-Max Scaling
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
#
#     ax1.hist(x, bins=20, color='skyblue', edgecolor='black')
#     ax1.set_title(feature + ' histo')
#
#     ax2.hist(scaledX, bins=20, color='lightgreen', edgecolor='black')
#     ax2.set_title('[0,1] scaled ' + feature + ' histo')
#
#     plt.show()


# pb1('Salary')
# pb1('Bonus %')
# pb1('Team')


def pb2(img):
    """
    Normalizes the pixel values of an image and displays the normalized image.

    Specifications:
    1. Opens an image located in the 'data/images' directory.
    2. Converts the image to a NumPy array.
    3. Normalizes the pixel values of the image array.
    4. Converts the normalized array back to an image.
    5. Displays the normalized image.

    Parameters:
    img (str): The filename of the image to be processed.

    Dependencies:
    - PIL: for image manipulation.
    - numpy: for numerical operations.

    Returns:
    None
    """
    image = Image.open('data/images/' + img)
    image_array = np.array(image)

    m = sum(image_array) / len(image_array)
    s = (1 / len(image_array) * sum([(i - m) ** 2 for i in image_array])) ** 0.5
    normalized_image = [(i - m) / s for i in image_array]
    # normalized_image = [(i - min(image_array)) / (max(image_array) - min(image_array)) for i in image_array]

    normalized_image = np.array(normalized_image)
    # normalized_image = normalized_image.reshape(image_array.shape)

    image = Image.fromarray((normalized_image * 255).astype(np.uint8))
    image.show()


# pb2('chatGPT.png')


def pb3():
    """
    Analyzes the frequency of words in each sentence of a text file.

    Specifications:
    1. Opens and reads the contents of a text file named 'texts.txt' in the 'data' directory.
    2. Tokenizes the text into sentences.
    3. For each sentence, tokenizes it into words and calculates the frequency of each word.
    4. Prints the word frequency for each sentence and displays a bar chart showing the frequency distribution.
    5. Increments the sentence index for each iteration.

    Dependencies:
    - nltk.tokenize.sent_tokenize: for sentence tokenization.
    - nltk.tokenize.word_tokenize: for word tokenization.
    - matplotlib.pyplot: for data visualization.

    Returns:
    None
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = sent_tokenize(text)

    i = 0
    for sentence in sentences:
        words = [word for word in word_tokenize(sentence) if word not in ['.', ',', '!', ':', '?', '"', '”'] and word not in '1234567890']
        word_counts = {word: words.count(word) for word in set(words)}

        print('Propozitia ' + str(i) + ': ')
        print(word_counts)

        plt.figure(figsize=(10, 5))
        plt.bar(word_counts.keys(), word_counts.values())
        plt.xlabel('Cuvânt')
        plt.ylabel('Număr de apariții')
        plt.title('Numărul de apariții al fiecărui cuvânt în propoziție')
        plt.xticks(rotation=45, ha='right')
        plt.show()

        i += 1
        print()


pb3()
