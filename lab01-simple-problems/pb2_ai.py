import math


# Complexity: O(1)
def pb2(punct1, punct2):
    x1, y1 = punct1
    x2, y2 = punct2
    distanta = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distanta


# punct1 = (1, 5)
# punct2 = (4, 1)
# print("Distanta euclidiana între", punct1, "și", punct2, "este:", distanta_euclidiana(punct1, punct2))

def tests():
    assert (pb2([4, 1], [1, 5]) == 5)
    assert (pb2([5, 0], [1, 0]) == 4)
    assert (pb2([0, 2], [0, 2]) == 0)
    assert (pb2([3, 0], [2, 0]) == 1)

tests()
