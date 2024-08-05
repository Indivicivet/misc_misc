import math
from collections import Counter

from tqdm import tqdm


def digits_to_power(v):
    """
    consider the map ...dbca -> ...d^4 c^3 b^2 a for ...dbca the decimal rep
    """
    return math.prod(int(c) ** (i + 1) for i, c in enumerate(str(v)[::-1]))


def iterated_digits_to_power(v0, n_iters=100):
    res = [v0]
    res_set = {v0}
    for _ in range(n_iters):
        iterated = digits_to_power(res[-1])
        if iterated in res_set:
            loop_circ = res[res.index(iterated):]
            lowest_idx = loop_circ.index(min(loop_circ))
            loop = loop_circ[lowest_idx:] + loop_circ[:lowest_idx]
            res.append(f"loop: {loop}")
            return res
        res.append(iterated)
        res_set.add(iterated)
    return res


def find_loops(max_val=10_000_000):
    outcomes = Counter()
    for i in tqdm(range(1, max_val)):
        result = iterated_digits_to_power(i)
        # print(i, result)
        outcomes[result[-1]] += 1
    return outcomes


def find_fx_eq_xplus1(max_val=10_000_000):
    matches = []
    for i in tqdm(range(1, max_val)):
        if digits_to_power(i) == i + 1:
            matches.append(i)
    return matches


if __name__ == "__main__":
    # print("loops:", find_loops())
    print("f(x) = x+1:", find_fx_eq_xplus1())
