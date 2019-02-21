def findGreatestDivisor(n):
    for divisor in range(n-1,0,-1):
        if n%divisor == 0:
            return divisor


print(findGreatestDivisor(10))
