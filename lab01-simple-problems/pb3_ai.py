# Complexity: O(n)
def pb3(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectorii trebuie să aibă aceeași lungime")

    suma_produse = 0

    for i in range(len(vector1)):
        suma_produse += vector1[i] * vector2[i]

    return suma_produse


# vector1 = [1, 0, 2, 0, 3]
# vector2 = [1, 2, 0, 3, 1]
# print("Produsul scalar al vectorilor:", produs_scalar(vector1, vector2))

def tests():
    assert(pb3([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]) == 4)
    assert(pb3([1, 0, 0, 0, 3], [1, 2, 1, 3, 2]) == 7)
    assert(pb3([1, 0, 2, 0, 3], [1, 2, 1, 3, 3]) == 12)
    assert(pb3([0, 0, 0, 0, 0], [1, 2, 0, 3, 1]) == 0)

tests()
