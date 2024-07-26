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


COL_FAC = np.array([0.7, 1, 0.5])


def distances(i1d, axs):
    axs = list(axs)  # to index into 1D properly ^^;
    return (
        np.sum(i1d[..., axs] ** 2 * COL_FAC[axs], axis=1) ** 0.5
        / (1e-5 + np.sum(i1d ** 2 * COL_FAC, axis=1) ** 0.5)
    )


remapped = np.zeros_like(sorted_1dc_lookup)
remaining_img = np.ones((sorted_1dc_lookup.shape[0], 1), dtype=int)
remaining_cols = np.ones((sorted_1dc_lookup.shape[0], 1), dtype=int)
for axes, max_ratio in {
    (1, ): 0.1,
    (0, ): 0.1,
    (2, ): 0.1,
    (0, 1): 0.1,
    (1, 2): 0.1,
    (0, 2): 0.1,
}.items():
    n_points = int(4096 * 4096 * max_ratio)
    img_points = np.argsort(
        distances(im_1d_col * remaining_img, axes)
    )[-n_points:]
    col_points = np.argsort(
        distances(sorted_1dc_values * remaining_cols, axes)
    )[-n_points:]
    remapped[img_points] = sorted_1dc_lookup[col_points]
    remaining_img[img_points] -= 1
    remaining_cols[col_points] -= 1

assert (remaining_img >= 0).all()
assert (remaining_cols >= 0).all()

img_pts_left = np.sum(remaining_img)
cols_left = np.sum(remaining_cols)
print(f"{img_pts_left=}, {cols_left=}")
assert img_pts_left == cols_left

# final "greys" update; this time using dist vs black rather than white
img_points = np.argsort(
    np.sum(im_1d_col ** 2 * COL_FAC, axis=1) * remaining_img[..., 0]
)[-img_pts_left:]
col_points = np.argsort(
    np.sum(sorted_1dc_values ** 2 * COL_FAC, axis=1) * remaining_cols[..., 0]
)[-img_pts_left:]
remapped[img_points] = sorted_1dc_lookup[col_points]

result_im = Image.fromarray(remapped.reshape((4096, 4096, 3)))
result_im.show()
result_im.save("4k_bijection_result.png")

print("checking uniqueness is 16777216...")
print("unique:", len(np.unique(remapped, axis=0)))
