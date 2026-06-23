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
import numpy as np


def print_arr(arr):
    print(f"shape {arr.shape}")
    print(f"dtype {arr.dtype}")
    with np.printoptions(threshold=10_000, linewidth=120):
        print(arr)


if __name__ == "__main__":
    print_arr(np.load(sys.argv[1]))
