"""
more or less naive implementation of the obvious algorithm that constructs an
infinite set of primes (constructive version of Euclid's original proof)
"""

import random
import math


def is_probable_prime(n):
    if n < 2:
        return False
    # write n-1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in [1, n - 1]:
            continue
        skip_to_next_a = False
        for _ in range(s - 1):
            x = (x ** 2) % n
            if x == n - 1:
                skip_to_next_a = True
                break
        if skip_to_next_a:
            continue
        return False
    return True


def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        c = random.randrange(1, n - 1)
        sqmod = lambda a: (a ** 2 + c) % n
        x = random.randrange(0, n)
        y = x
        d = 1
        while d == 1:
            x = sqmod(x)
            y = sqmod(sqmod(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def smallest_prime_factor(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    # trial divide a bit to strip tiny factors quickly
    j = 5
    w = 2
    while j ** 2 <= n and j < 10_000:
        if n % j == 0:
            return j
        j += w
        w = 6 - w
    if is_probable_prime(n):
        return n
    # Pollard Rho factor, then recurse until prime
    f = pollard_rho(n)
    g = n // f
    return min(
        f if is_probable_prime(f) else smallest_prime_factor(f),
        g if is_probable_prime(g) else smallest_prime_factor(g),
    )


def get_next_prime(primes):
    prod = 1
    for p in primes:
        prod *= p
    return smallest_prime_factor(prod + 1)


if __name__ == "__main__":
    ps = [2]
    for i in range(100):  # above 51 not known :)
        print(f"#{i + 1}: {ps[-1]}")
        ps.append(get_next_prime(ps))
