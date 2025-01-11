"""
Extracts words from the input sentence that appear exactly once.

Args:
    sentence (str): A string representing the input sentence containing words separated by spaces.

Returns:
    list: A list of words that appear exactly once in the given sentence.

Complexity:
    O(n)
"""


def pb4(sentence):
    words = sentence.split(" ")
    appearances = {}
    one_appearance_words = []

    for word in words:
        appearances[word] = 0

    for word in words:
        appearances[word] += 1

    for word in words:
        if appearances[word] == 1:
            one_appearance_words.append(word)

    return one_appearance_words


def tests():
    assert(pb4("ana are ana are mere rosii ana") == ['mere', 'rosii'])
    assert(pb4("ana are ana are ana") == [])
    assert(pb4("ana are ana mere ana") == ['are', 'mere'])


tests()
