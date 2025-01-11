def cifra_minima(n):
    cifra_min = 9
    while n:
        cifra = n % 10
        cifra_min = min(cifra_min, cifra)
        n //= 10
    return cifra_min