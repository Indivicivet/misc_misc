from pathlib import Path

import numpy as np
from PIL import Image

im_path = (
    Path(__file__).parent
    / "easy-peasy-dot-ai__fc61fa66-850f-48d0-b287-2131be2d8880.png"
)
im = np.array(Image.open(im_path).convert("RGB").resize((4096, 4096)), dtype=float)
im_1d_col = im.reshape(4096 * 4096, 3)
sorted_1dc_lookup = np.array([
    (r, g, b)
    for r in range(256)
    for g in range(256)
    for b in range(256)
], dtype=np.uint8).reshape(4096 * 4096, 3)
sorted_1dc_values = sorted_1dc_lookup.astype(float)


def distances(i1d, axs):
    return (
        np.sum(i1d[..., axs] ** 2, axis=1) ** 0.5
        / (1e-5 + np.sum(i1d ** 2, axis=1) ** 0.5)
    )


PER_COL_N = 4096 * 4096 // 10

remapped = np.zeros_like(sorted_1dc_lookup)
hit_img = np.zeros_like(sorted_1dc_lookup[..., 0])
hit_cols = np.zeros_like(sorted_1dc_lookup[..., 0])
for axes in [[1], [0], [2], [0, 1], [1, 2], [0, 2]]:
    img_points = np.argsort(distances(im_1d_col, axes))[-PER_COL_N:]
    col_points = np.argsort(distances(sorted_1dc_values, axes))[-PER_COL_N:]
    # todo :: actually need to just not update points that were already hit
    hit_img[img_points] += 1
    hit_cols[col_points] += 1
    remapped[img_points] = sorted_1dc_lookup[col_points]

# mishit_img = hit_img != 1
# mishit_cols = hit_cols != 1
# assert np.sum(mishit_img) == np.sum(mishit_cols)

result_im = Image.fromarray(remapped.reshape((4096, 4096, 3)))
result_im.show()
result_im.save("4k_bijection_result.png")

print("checking uniqueness is 16777216...")
print("unique:", len(np.unique(remapped, axis=0)))
