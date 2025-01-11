def maxim_lista(lista):
    maxim = lista[-1]
    for el in lista:
        if maxim < el:
            maxim = el
    return maxim