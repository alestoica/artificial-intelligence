def mediana(lista):
    sorted_lista = sorted(lista)
    lungime = len(sorted_lista)
    if lungime % 2 == 0:
        return (sorted_lista[lungime//2 - 1] + sorted_lista[lungime//2]) / 2
    else:
        return sorted_lista[lungime//2]