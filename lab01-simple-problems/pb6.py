"""
Finds the majority element in a given list.

Args:
    vect (list): A list of integers.

Returns:
    int or str: The majority element if it exists, otherwise a message indicating no majority element.

Complexity:
    O(n)
"""


def pb6(vect):
    vect.sort()
    maj_elem = vect[0]
    maxim = 0

    for el in vect:
        if el != maj_elem:
            if maxim >= len(vect) / 2:
                return maj_elem
            maj_elem = el
            maxim = 0

        if el == maj_elem:
            maxim += 1

    return "There is no majority element!"


def tests():
    assert (pb6([2, 8, 7, 5, 3, 1, 2]) == "There is not majority element!")
    assert (pb6([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2)
    assert (pb6([2, 8, 3, 7, 3, 3, 1, 3]) == 3)


tests()
