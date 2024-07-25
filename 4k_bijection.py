from pathlib import Path

import numpy as np
from PIL import Image

im_path = (
    Path(__file__).parent
    / "easy-peasy-dot-ai__fc61fa66-850f-48d0-b287-2131be2d8880.png"
)
im = np.array(Image.open(im_path).convert("RGB").resize((4096, 4096)))
im_1d_col = im.reshape(4096 * 4096, 3)
sorted_1dc = np.array([
    (r, g, b)
    for r in range(256)
    for g in range(256)
    for b in range(256)
], dtype=np.uint8).reshape(4096 * 4096, 3)


def distances(i1d, axis):
    return i1d[..., axis] / (1e-5 + np.sum(i1d ** 2, axis=1) ** 0.5)


def invsort(arr):
    argsort = np.argsort(arr)
    inv = np.arange(len(argsort))
    inv[argsort] = inv.copy()
    return inv


im_proj_green = distances(im_1d_col, 1)
sorted_1dc_g = distances(sorted_1dc, 1)

green_invsort = invsort(im_proj_green)
allcol_sort = np.argsort(sorted_1dc_g)
remapped = sorted_1dc[allcol_sort][green_invsort]


Image.fromarray(remapped.reshape(4096, 4096, 3)).show()

print("checking uniqueness is 16777216...")
print("unique:", len(np.unique(remapped, axis=0)))
