# Complexity: O(n)
def pb5(lista):
    frecventa = {}
    for numar in lista:
        if numar in frecventa:
            return numar
        else:
            frecventa[numar] = 1


# sir = [1, 2, 3, 4, 2]
# valoare_repetata = gaseste_repetare(sir)
# print("Valoarea care se repetă de două ori în șir este:", valoare_repetata)

def tests():
    assert(pb5([1, 2, 3, 4, 2]) == 2)
    assert(pb5([1, 1, 2]) == 1)
    assert(pb5([1, 1]) == 1)
    assert(pb5([1, 2, 3, 4, 4, 5]) == 4)
    assert(pb5([1, 2, 3, 3, 4, 5, 6]) == 3)

tests()
