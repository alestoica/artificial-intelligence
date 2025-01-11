"""
Finds the index of the row with the maximum sum of elements in the given matrix.

Parameters:
    matrix (list of lists): 2D matrix represented as a list of lists.

Returns:
    int: Index of the row with the maximum sum of elements.
         Returns -1 if the matrix is empty.

Complexity:
    O(m * n)
"""


def pb10(matrix):
    maximum = 0
    max_line = -1
    pos = 0

    for vect in matrix:
        if sum(vect) > maximum:
            maximum = sum(vect)
            max_line = pos

        pos += 1

    return max_line


def tests():
    matrix = [[0, 0, 0, 1, 1],
              [0, 1, 1, 1, 1],
              [0, 0, 1, 1, 1]]
    assert(pb10(matrix) == 1)

    matrix = [[1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1],
              [0, 0, 1, 1, 1]]
    assert(pb10(matrix) == 0)

    matrix = [[0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 1, 1, 1]]
    assert(pb10(matrix) == 2)


tests()
