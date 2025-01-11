# Complexity: O(nlog(n))
def pb8(n):
    numere_bianre = []
    for i in range(1, n + 1):
        numere_bianre.append(bin(i)[2:])
    return numere_bianre


# n = 4
# rezultat = generare_numere_binare(n)
# print("Numerele binare generate între 1 și", n, "sunt:")
# for numar in rezultat:
#     print(numar)


def tests():
    try:
        assert(pb8(4) == [1, 10, 11, 100])
        assert(pb8(5) == [1, 10, 11, 100, 101])
        assert(pb8(8) == [1, 10, 11, 100, 101, 110, 111, 1000])
        assert(pb8(10) == [1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010])
    except:
        print("An error occurred!")

tests()
