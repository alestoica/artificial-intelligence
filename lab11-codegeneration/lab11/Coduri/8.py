def cifra_maxima(n):
    cifra_max = 0
    while n:
        cifra = n % 10
        cifra_max = max(cifra_max, cifra)
        n //= 10
    return cifra_max