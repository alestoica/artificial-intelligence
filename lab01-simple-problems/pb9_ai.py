# Complexity: O(k * (r - p + 1) * (s - q + 1))
def pb9(matrice, liste_perechi):
    rezultate = []
    for pereche in liste_perechi:
        (p, q), (r, s) = pereche
        suma_submatrice = 0
        for i in range(p, r + 1):
            for j in range(q, s + 1):
                suma_submatrice += matrice[i][j]
        rezultate.append(suma_submatrice)
    return rezultate


# matrice = [
#     [0, 2, 5, 4, 1],
#     [4, 8, 2, 3, 7],
#     [6, 3, 4, 6, 2],
#     [7, 3, 1, 8, 3],
#     [1, 5, 7, 9, 4]
# ]
# liste_perechi = [((1, 1), (3, 3)), ((2, 2), (4, 4))]
#
# rezultate = calculeaza_sume(matrice, liste_perechi)
# for i, suma in enumerate(rezultate, start=1):
#     print(f'Suma pentru perechea {i}: {suma}')


def tests():
    try:
        matrix = [[0, 2, 5, 4, 1],
                  [4, 8, 2, 3, 7],
                  [6, 3, 4, 6, 2],
                  [7, 3, 1, 8, 3],
                  [1, 5, 7, 9, 4]]
        assert(pb9(matrix, [[[1, 1], [3, 3]], [[2, 2], [4, 4]]]) == [38, 44])
        assert(pb9(matrix, [[[1, 1], [2, 2]], [[0, 0], [1, 1]]]) == [17, 14])
        assert(pb9(matrix, [[[2, 3], [3, 5]], [[3, 4], [4, 6]]]) == [19, 7])
    except:
        print("An error occurred!")


tests()
