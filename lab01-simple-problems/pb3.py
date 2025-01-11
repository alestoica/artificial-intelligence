"""
Computes the dot product of two vectors.

Args:
    v1 (list): First vector.
    v2 (list): Second vector.

Returns:
    int: The dot product of the two vectors.

Complexity:
    O(n)
"""


def pb3(v1, v2):
    s = 0
    length = len(v1)
    for pos in range(length):
        s += v1[pos] * v2[pos]
    return s


def tests():
    assert(pb3([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]) == 4)
    assert(pb3([1, 0, 0, 0, 3], [1, 2, 1, 3, 2]) == 7)
    assert(pb3([1, 0, 2, 0, 3], [1, 2, 1, 3, 3]) == 12)
    assert(pb3([0, 0, 0, 0, 0], [1, 2, 0, 3, 1]) == 0)


tests()
