"""
Finds the k-th largest element in a list.

Args:
    vect (listint): The input list of integers.
    k (int): The position of the element to find (1-indexed).

Returns:
    int: The k-th largest element in the list.

Complexity:
    O(nlog(n))
"""


def pb7(vect, k):
    vect.sort()
    return vect[-k]


def tests():
    assert(pb7([7, 4, 6, 3, 9, 1], 2) == 7)
    assert (pb7([7, 4, 6, 3, 9, 1], 3) == 6)
    assert (pb7([7, 4, 6, 3, 9, 1], 6) == 1)


tests()
