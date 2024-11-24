"""
3x3 patterns mod symmetry
"""

import itertools

import numpy as np

SIZE = 3
DISPLAY_ROW_WIDTH = 10

bit_vals = (1 << np.arange(SIZE ** 2 - 1, -1, -1)).reshape((SIZE, SIZE))

results = []
already_hit = set()
for bin_packed in range(2 ** (SIZE ** 2)):
    if bin_packed in already_hit:
        continue
    results.append(bin_packed)
    unpacked = (bin_packed & bit_vals).astype(np.bool)
    for arr in [
        unpacked,
        unpacked[::-1, :],
        unpacked[:, ::-1],
        unpacked[::-1, ::-1],
        unpacked.T,
        unpacked.T[::-1, :],
        unpacked.T[:, ::-1],
        unpacked.T[::-1, ::-1],
    ]:
        already_hit.add(np.sum(arr * bit_vals))

for row in itertools.batched(results, DISPLAY_ROW_WIDTH):
    arrs = [(arr & bit_vals).astype(np.bool) for arr in row]
    print("-" * (4 * DISPLAY_ROW_WIDTH - 1))
    for jj in range(SIZE):
        print(
            "|".join(
                "".join("X" if val else " " for val in arr[jj])
                for arr in arrs
            )
        )
