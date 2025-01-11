import math

import matplotlib
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize
matplotlib.use('TkAgg')


def calculate_word_counts(sentence):
    """
    Calculates the frequency of each word in a sentence.

    Parameters:
    sentence (str): The input sentence.

    Returns:
    dict: A dictionary containing the word frequencies.
    """
    words = [word for word in word_tokenize(sentence) if
             word not in ['.', ',', '!', ':', '?', '"', '”'] and word not in '1234567890']
    word_counts = {word: words.count(word) for word in set(words)}

    return word_counts


def normalize_word_counts(sentence):
    """
    Normalizes the word frequencies in a sentence using log transformation.

    Parameters:
    sentence (str): The input sentence.

    Returns:
    list: A list of normalized word frequencies.
    """
    word_counts = calculate_word_counts(sentence)

    normalized_counts = {word: math.log(count + 1) for word, count in word_counts.items()}

    return list(normalized_counts.values())


def normalize_word_counts_in_sentences():
    """
    Normalizes the word frequencies in sentences and visualizes the distribution.

    Specifications:
    1. Opens and reads the contents of a text file named 'texts.txt' in the 'data' directory.
    2. Tokenizes the text into sentences.
    3. Calculates the initial and normalized word frequencies for each sentence.
    4. Displays histograms for the initial and normalized word frequencies.

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

    initial_counts = []
    normalized_counts = []
    for sentence in sentences:
        initial_counts.extend(list(calculate_word_counts(sentence).values()))
        normalized_counts.extend(normalize_word_counts(sentence))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.hist(initial_counts, bins=5, color='blue', alpha=0.7)
    ax1.set_title('Datele initiale')
    ax1.set_xlabel('Frecvență cuvinte')
    ax1.set_ylabel('Număr de cuvinte')

    ax2.hist(normalized_counts, bins=5, color='green', alpha=0.7)
    ax2.set_title('Datele normalizate')
    ax2.set_xlabel('Frecvență cuvinte (normalizate)')
    ax2.set_ylabel('Număr de cuvinte')

    plt.tight_layout()
    plt.show()


normalize_word_counts_in_sentences()
