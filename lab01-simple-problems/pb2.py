from cmath import sqrt

"""
Computes the Euclidean distance between two points in a 2D plane.

Parameters:
    point1 (list): A list representing the coordinates of the first point [x1, y1].
    point2 (list): A list representing the coordinates of the second point [x2, y2].

Returns:
    float: The Euclidean distance between the two points.
    
Complexity:
    O(1)
"""


def pb2(point1, point2):
    return sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))


def tests():
    assert (pb2([4, 1], [1, 5]) == 5)
    assert (pb2([5, 0], [1, 0]) == 4)
    assert (pb2([0, 2], [0, 2]) == 0)
    assert (pb2([3, 0], [2, 0]) == 1)


tests()
