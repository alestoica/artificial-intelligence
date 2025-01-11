import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from unidecode import unidecode


def number_of_sentences():
    """
    Counts the number of sentences in a text file.

    Specifications:
    1. Opens and reads the contents of a text file named 'texts.txt' in the 'data' directory.
    2. Tokenizes the text into sentences.
    3. Prints the number of sentences.
    4. Returns the number of sentences.

    Dependencies:
    - nltk.tokenize.sent_tokenize: for sentence tokenization.

    Returns:
    int: The number of sentences in the text file.
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = sent_tokenize(text)

    print('Numarul de propozitii este: ', len(sentences))
    print()
    return len(sentences)


def number_of_words():
    """
    Counts the number of words in a text file.

    Specifications:
    1. Opens and reads the contents of a text file named 'texts.txt' in the 'data' directory.
    2. Tokenizes the text into words.
    3. Prints the total number of words and the number of unique words.
    4. Returns a list containing the total number of words and the number of unique words.

    Dependencies:
    - nltk.tokenize.word_tokenize: for word tokenization.

    Returns:
    list: [total_number_of_words, number_of_unique_words]
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    words = word_tokenize(text)

    print('Numarul de cuvinte este: ', len(words))
    print()
    print('Numarul de cuvinte diferite este: ', len(set(words)))
    print()
    return [len(words), len(set(words))]


def longest_shortest_words():
    """
    Finds the longest and shortest words in a text file.

    Specifications:
    1. Opens and reads the contents of a text file named 'texts.txt' in the 'data' directory.
    2. Tokenizes the text into words.
    3. Finds the longest and shortest words, excluding punctuation marks and digits.
    4. Prints the sets of longest and shortest words.
    5. Returns a list containing sets of longest and shortest words.

    Dependencies:
    - nltk.tokenize.word_tokenize: for word tokenization.

    Returns:
    list: [set_of_longest_words, set_of_shortest_words]
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    words = word_tokenize(text)

    max_length = max(len(word) for word in words)
    min_length = min(len(word) for word in words)

    longest_words = [word for word in words if len(word) == max_length]
    shortest_words = [word for word in words if len(word) == min_length and word not in ['.', ',', '!', ':', '?', '"', '”'] and word not in '1234567890']

    print('Cele mai lungi cuvinte: ', set(longest_words))
    print('Cele mai scurte cuvinte: ', set(shortest_words))
    print()
    return [set(longest_words), set(shortest_words)]


def without_diacritics():
    """
    Removes diacritics from the text in a text file.

    Specifications:
    1. Opens and reads the contents of a text file named 'texts.txt' in the 'data' directory.
    2. Removes diacritics from the text.
    3. Prints the text without diacritics.
    4. Returns the text without diacritics.

    Dependencies:
    - unidecode.unidecode: for removing diacritics.

    Returns:
    str: The text without diacritics.
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    text_without_diacritics = unidecode(text)

    print('Textul fara diacritice este: \n', text_without_diacritics)
    return text_without_diacritics


def find_synonyms(word):
    """
    Finds synonyms of a word using WordNet.

    Specifications:
    1. Finds synonyms of a given word in Romanian using WordNet.
    2. Prints the set of synonyms.
    3. Returns the set of synonyms.

    Dependencies:
    - nltk.corpus.wordnet: for synonym lookup.

    Parameters:
    word (str): The word for which synonyms are to be found.

    Returns:
    set: Set of synonyms of the given word.
    """
    synonyms = []

    for syn in wordnet.synsets(word, lang='ron'):
        for lemma in syn.lemmas(lang='ron'):
            synonyms.append(lemma.name())

    print('Sinonimele cuvantului ', word, ' sunt: ', set(synonyms))
    print()
    return set(synonyms)


# number_of_sentences()
# number_of_words()
# longest_shortest_words()
# without_diacritics()
#
# print()

# for w in set(longest_shortest_words()):
#     find_synonyms(w)

# find_synonyms('laborator')


def test_pb3():
    assert (number_of_sentences() == 10)
    assert (number_of_words()[0] == 182)
    assert (number_of_words()[1] == 99)
    assert (longest_shortest_words()[0] == {'laboratoarele'})
    assert (longest_shortest_words()[1] == {'e', 'o'})
    assert (find_synonyms('laborator') == {'laborator', 'laboratordeștiințe', 'poligondeîncercare', 'laboratordecercetare'})


test_pb3()
