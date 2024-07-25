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


PER_COL_N = 4096 * 4096 // 10

remapped = np.zeros_like(sorted_1dc)
for ax in [1, 0, 2]:
    remapped[
        np.argsort(distances(im_1d_col, ax))[-PER_COL_N:]
    ] = sorted_1dc[np.argsort(distances(sorted_1dc, ax))[-PER_COL_N:]]

result_im = Image.fromarray(remapped.reshape((4096, 4096, 3)))
result_im.show()
result_im.save("4k_bijection_result.png")

print("checking uniqueness is 16777216...")
print("unique:", len(np.unique(remapped, axis=0)))
