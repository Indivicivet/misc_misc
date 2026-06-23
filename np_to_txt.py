r"""
.npy to text intended for diffing .npy files

git config --global diff.np_to_txt.textconv "py <path to>/np_to_txt.py"

add to gitattributes:
*.npy diff=np_to_txt
e.g. windows default:
%USERPROFILE%\.config\git\attributes
or get path via
git config --global core.attributesFile
"""

import sys
import hashlib

import numpy as np


def print_arr(arr):
    print(f"shape {arr.shape}")
    print(f"dtype {arr.dtype}")
    print(f"min {arr.min()} max {arr.max()}")
    print(f"hash {hashlib.sha256(arr.tobytes()).hexdigest()[:20]}")
    print()
    with np.printoptions(
        threshold=10_000,  # truncate if >this many elems
        edgeitems=16,  # rows and columns for truncated
        linewidth=160,
    ):
        print(arr)


if __name__ == "__main__":
    print_arr(np.load(sys.argv[1]))
