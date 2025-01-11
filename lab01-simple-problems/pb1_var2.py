"""
This function sorts a given list/array using the Bubble Sort algorithm.

Parameters:
    arr (list): The list to be sorted.

Returns:
    list: The sorted list.

Complexity: O(n^2)
"""


def bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


def pb1(sentence):
    words = sentence.split(" ")
    return bubble_sort(words)[-1]


def tests():
    assert (pb1('Ana are mere si portocale') == 'si')
    assert (pb1('A a a a') == 'a')
    assert (pb1('a z c y') == 'z')
    assert (pb1('Zonia are mere') == 'mere')
    assert (pb1('Gigel merge la piata') == 'piata')


tests()