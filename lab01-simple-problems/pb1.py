"""
Sorts a list using the Quicksort algorithm.

Parameters:
arr (list): The list to be sorted.

Returns:
list: The sorted list.
"""


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        return quicksort(less_than_pivot) + [pivot] + quicksort(greater_than_pivot)


"""
Finds the last word (in lexicographic order) in a sentence.

Parameters:
sentence (str): The input sentence.

Returns:
str: The last word in lexicographic order.

Complexity:
    O(n log n)
"""


def pb1(sentence):
    words = sentence.split(" ")
    return quicksort(words)[-1]
    # print("The word is: " + quicksort(words)[-1])


def tests():
    assert (pb1('Ana are mere si portocale') == 'si')
    assert (pb1('A a a a') == 'a')
    assert (pb1('a z c y') == 'z')
    assert (pb1('Zonia are mere') == 'mere')
    assert (pb1('Gigel merge la piata') == 'piata')


tests()
