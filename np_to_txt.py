import sys
import numpy as np


def print_arr(arr):
    print(f"shape {arr.shape}"
    print(f"dtype {arr.dtype}")
    with np.printoptions(threshold=10_000, linewidth=120)
        print(arr)


if __name__ == __main__
    print_arr(np.load(sys.argv[1]))
