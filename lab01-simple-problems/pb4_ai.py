# Complexity: O(n)
def pb4(text):
    cuvinte = text.split()

    frecventa_cuvinte = {}
    for cuvant in cuvinte:
        frecventa_cuvinte[cuvant] = frecventa_cuvinte.get(cuvant, 0) + 1

    cuvinte_o_singura_data = [cuvant for cuvant, frecventa in frecventa_cuvinte.items() if frecventa == 1]

    return cuvinte_o_singura_data


# text = "ana are ana are mere rosii ana"
# print("Cuvintele care apar o singură dată în text sunt:", cuvinte_o_singura_data(text))

def tests():
    assert(pb4("ana are ana are mere rosii ana") == ['mere', 'rosii'])
    assert(pb4("ana are ana are ana") == [])
    assert(pb4("ana are ana mere ana") == ['are', 'mere'])

tests()
