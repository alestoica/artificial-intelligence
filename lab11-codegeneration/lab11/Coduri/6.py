def suma_cifre(n):
    suma = 0
    while n:
        suma += n % 10
        n //= 10
    return suma