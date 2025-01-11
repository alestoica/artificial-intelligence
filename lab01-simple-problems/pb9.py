"""
Computes the sum of elements within specified submatrices.

Parameters:
    matrix (list of lists): The matrix containing integer elements.
    coords (list of lists of lists): Coordinates of the submatrices to compute the sum.

Returns:
    list: List containing the sum of elements for each specified sub-matrix.

Complexity:
    O(m*n*k)
"""
def pb9(matrix, coords):
    result = []

    for pair in coords:
        start_point = pair[0]
        end_point = pair[1]
        sum_elem = 0

        for vect in matrix[start_point[0]:end_point[0] + 1]:
            sum_elem += sum(vect[start_point[1]:end_point[1] + 1])
        result.append(sum_elem)

    return result


def tests():
    matrix = [[0, 2, 5, 4, 1],
              [4, 8, 2, 3, 7],
              [6, 3, 4, 6, 2],
              [7, 3, 1, 8, 3],
              [1, 5, 7, 9, 4]]
    assert(pb9(matrix, [[[1, 1], [3, 3]], [[2, 2], [4, 4]]]) == [38, 44])
    assert(pb9(matrix, [[[1, 1], [2, 2]], [[0, 0], [1, 1]]]) == [17, 14])
    assert(pb9(matrix, [[[2, 3], [3, 5]], [[3, 4], [4, 6]]]) == [19, 7])


tests()
