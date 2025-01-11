"""
Generates the binary representation of integers from 1 to n (inclusive).

Parameters:
    n (int): Upper limit (inclusive) for generating binary representations.

Returns:
    list: List containing binary representations of integers from 1 to n.

Complexity:
    O(nlog(n))
"""


def pb8(n):
    bin_rep = []

    for el in range(1, n + 1):
        i = el
        power = 1
        number = 0

        while i >= 1:
            number = number + power * int(i % 2)
            power *= 10
            i = int(i / 2)

        bin_rep.append(number)

    return bin_rep


def tests():
    assert(pb8(4) == [1, 10, 11, 100])
    assert(pb8(5) == [1, 10, 11, 100, 101])
    assert(pb8(8) == [1, 10, 11, 100, 101, 110, 111, 1000])
    assert(pb8(10) == [1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010])


tests()
