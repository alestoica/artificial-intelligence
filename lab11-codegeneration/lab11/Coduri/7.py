def inverseaza_cifrele(n):
    invers = 0
    while n:
        cifra = n % 10
        invers = invers * 10 + cifra
        n //= 10
    return invers