"""
naive implementation of the obvious algorithm that constructs an
infinite set of primes (constructive version of Euclid's original proof)
"""

N = 30


def get_next_prime(primes):
    prod = 1
    for p in primes:
        prod *= p
    q = prod + 1
    for p_candidate in range(2, q + 1):
        if q % p_candidate == 0:
            return p_candidate
    assert False, "q should factor q..."


if __name__ == "__main__":
    ps = [2]
    for i in range(N):
        print(f"#{i+1}: {ps[-1]}")
        ps.append(get_next_prime(ps))
