def digits_to_power(v):
    """
    consider the map ...dbca -> ...d^4 c^3 b^2 a for ...dbca the decimal rep
    """
    return math.prod(int(c) ** (i + 1) for i, c in enumerate(str(v)[::-1]))


def iterated_digits_to_power(v0, n_iters=10):
    res = [v0]
    res_set = {v0}
    for _ in range(n_iters):
        iterated = digits_to_power(res[-1])
        if iterated in res_set:
            res.append(f"looped {iterated}")
            return res
        res.append(iterated)
        res_set.add(iterated)
    return res


if __name__ == "__main__":
    print(iterated_digits_to_power(345))
