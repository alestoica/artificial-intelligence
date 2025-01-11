# Complexity: O(n * (log(m) + m))
def pb10(matrice):
    n = len(matrice)
    m = len(matrice[0])

    linia_maxim_1 = 0
    max_1 = 0

    for i in range(n):
        stanga = 0
        dreapta = m - 1
        while stanga <= dreapta:
            mijloc = (stanga + dreapta) // 2
            if matrice[i][mijloc] == 1:
                max_1_linie_curenta = m - mijloc
                if max_1_linie_curenta > max_1:
                    max_1 = max_1_linie_curenta
                    linia_maxim_1 = i
                break
            else:
                stanga = mijloc + 1

    return linia_maxim_1


# matrice = [
#     [0, 0, 0, 1, 1],
#     [0, 1, 1, 1, 1],
#     [0, 0, 1, 1, 1]
# ]
# index_linie_maxim_1 = identificare_linie_maxim_1(matrice)
# print("Indexul liniei cu cele mai multe elemente de 1:", index_linie_maxim_1)


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
