import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb, rgb2lab, rgb2xyz, deltaE_ciede2000
from tqdm import tqdm

TIMES_PRIS = 6
DELTA_EQS = [1, 2]
grid_size = 500
h_vals = np.linspace(0, 1, grid_size)
s_vals = np.linspace(0.01, 1, grid_size)
H, S = np.meshgrid(h_vals, s_vals)
V = np.ones_like(H)

hsv_grid = np.dstack((H, S, V))
rgb_grid = hsv2rgb(hsv_grid)
lab_grid = rgb2lab(rgb_grid)
xyz_grid = rgb2xyz(rgb_grid)

X = xyz_grid[..., 0]
Y = xyz_grid[..., 1]
Z = xyz_grid[..., 2]

denominator = X + 15 * Y + 3 * Z
u_prime = (4 * X) / denominator
v_prime = (9 * Y) / denominator

centers_hsv = [
    (t, sat, 1)
    for t in np.linspace(0, 1, 3 * TIMES_PRIS + 1)[:-1]
    for sat in np.linspace(0, 1, 7) ** 1.5
]

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121, projection="polar")
ax1.pcolormesh(H * 2 * np.pi, S, rgb_grid, shading="gouraud", alpha=0.3)
ax1.set_yticks([])
ax1.set_xticks(np.linspace(0, 2 * np.pi, 6, endpoint=False))
ax1.set_xticklabels(["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"])
ax1.set_title("CIEDE2000 (ΔE=2.5) in sRGB HS Space")

ax2 = fig.add_subplot(122)
ax2.plot(
    np.append(u_prime[-1, :], u_prime[-1, 0]),
    np.append(v_prime[-1, :], v_prime[-1, 0]),
    color="gray",
    linestyle="--",
    label="sRGB Gamut Boundary"
)
ax2.set_xlabel("u'")
ax2.set_ylabel("v'")
ax2.set_title(f"CIEDE2000 (ΔE={DELTA_EQS}) in CIE 1976 UCS")
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_aspect("equal", adjustable="datalim")

for ch, cs, cv in tqdm(centers_hsv):
    center_rgb = hsv2rgb(np.array([[[ch, cs, cv]]]))
    center_lab = rgb2lab(center_rgb)
    center_xyz = rgb2xyz(center_rgb)

    delta_e_grid = deltaE_ciede2000(center_lab, lab_grid)

    ax1.contour(
        H * 2 * np.pi,
        S,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    ax1.plot(ch * 2 * np.pi, cs, "ko", markersize=3)

    ax2.contour(
        u_prime,
        v_prime,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    c_x, c_y, c_z = center_xyz[0, 0]
    c_den = c_x + 15 * c_y + 3 * c_z
    c_u = (4 * c_x) / c_den
    c_v = (9 * c_y) / c_den
    ax2.plot(c_u, c_v, "ko", markersize=3)

plt.tight_layout()
plt.show()
