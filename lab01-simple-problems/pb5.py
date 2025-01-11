"""
Find the value that appears exactly twice in the input list.

Args:
    vect (list): The input list containing values from the set {1, 2, ..., n - 1}
                 where only one value is repeated twice.

Returns:
    int: The value that appears exactly twice in the input list.

Complexity:
    O(n)
"""


def pb5(vect):
    appearances = len(vect) * [0]
    for el in vect:
        appearances[el] += 1

    return [el for el in vect if appearances[el] == 2][0]


def tests():
    assert(pb5([1, 2, 3, 4, 2]) == 2)
    assert(pb5([1, 1, 2]) == 1)
    assert(pb5([1, 1]) == 1)
    assert(pb5([1, 2, 3, 4, 4, 5]) == 4)
    assert(pb5([1, 2, 3, 3, 4, 5, 6]) == 3)


tests()
