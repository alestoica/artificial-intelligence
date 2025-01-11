def pb1(text):
    cuvinte = text.split(" ")
    cuvinte.sort()
    return cuvinte[-1]


def tests():
    assert (pb1('Ana are mere si portocale') == 'si')
    assert (pb1('A a a a') == 'a')
    assert (pb1('a z c y') == 'z')
    assert (pb1('Zonia are mere') == 'mere')
    assert (pb1('Gigel merge la piata') == 'piata')


tests()
