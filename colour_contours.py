import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb, rgb2lab, rgb2xyz, deltaE_ciede2000
from tqdm import tqdm
import colour

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

denominator_ucs = X + 15 * Y + 3 * Z
u_prime = (4 * X) / denominator_ucs
v_prime = (9 * Y) / denominator_ucs

denominator_xyz = X + Y + Z
# Add an epsilon to prevent division by zero in case of black/zero
x_prime = X / np.maximum(denominator_xyz, 1e-10)
y_prime = Y / np.maximum(denominator_xyz, 1e-10)

# Calculate coordinates for CAM16-UCS
cam16_grid = colour.XYZ_to_CAM16UCS(xyz_grid)
a_cam16 = cam16_grid[..., 1]
b_cam16 = cam16_grid[..., 2]

# Calculate coordinates for DIN99
din99_grid = colour.XYZ_to_DIN99(xyz_grid)
a_din99 = din99_grid[..., 1]
b_din99 = din99_grid[..., 2]

# Calculate coordinates for IPT
ipt_grid = colour.XYZ_to_IPT(xyz_grid)
P_ipt = ipt_grid[..., 1]
T_ipt = ipt_grid[..., 2]

centers_hsv = [
    (t, sat, 1)
    for t in np.linspace(0, 1, 3 * TIMES_PRIS + 1)[:-1]
    for sat in np.linspace(0, 1, 7) ** 1.5
]

fig = plt.figure(figsize=(20, 12))

# --- Row 1 ---
# Polar space
ax1 = fig.add_subplot(231, projection="polar")
ax1.pcolormesh(H * 2 * np.pi, S, rgb_grid, shading="gouraud", alpha=0.3)
ax1.set_yticks([])
ax1.set_xticks(np.linspace(0, 2 * np.pi, 6, endpoint=False))
ax1.set_xticklabels(["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"])
ax1.set_title("CIEDE2000 (ΔE=2.5) in sRGB HS Space")

# CIE 1931 xy
ax2 = fig.add_subplot(232)
ax2.pcolormesh(x_prime, y_prime, rgb_grid, shading="gouraud", alpha=0.3)
ax2.plot(
    np.append(x_prime[-1, :], x_prime[-1, 0]),
    np.append(y_prime[-1, :], y_prime[-1, 0]),
    color="gray",
    linestyle="--",
    label="sRGB Gamut Boundary",
)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"CIEDE2000 (ΔE={DELTA_EQS}) in CIE 1931 xy")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")
ax2.set_aspect("equal", adjustable="datalim")

# CIE 1976 UCS
ax3 = fig.add_subplot(233)
ax3.pcolormesh(u_prime, v_prime, rgb_grid, shading="gouraud", alpha=0.3)
ax3.plot(
    np.append(u_prime[-1, :], u_prime[-1, 0]),
    np.append(v_prime[-1, :], v_prime[-1, 0]),
    color="gray",
    linestyle="--",
    label="sRGB Gamut Boundary",
)
ax3.set_xlabel("u'")
ax3.set_ylabel("v'")
ax3.set_title(f"CIEDE2000 (ΔE={DELTA_EQS}) in CIE 1976 UCS")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right")
ax3.set_aspect("equal", adjustable="datalim")

# --- Row 2 ---
# CAM16-UCS
ax4 = fig.add_subplot(234)
ax4.pcolormesh(a_cam16, b_cam16, rgb_grid, shading="gouraud", alpha=0.3)
ax4.plot(
    np.append(a_cam16[-1, :], a_cam16[-1, 0]),
    np.append(b_cam16[-1, :], b_cam16[-1, 0]),
    color="gray",
    linestyle="--",
    label="sRGB Gamut Boundary",
)
ax4.set_xlabel("a'")
ax4.set_ylabel("b'")
ax4.set_title(f"CIEDE2000 (ΔE={DELTA_EQS}) in CAM16-UCS")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper right")
ax4.set_aspect("equal", adjustable="datalim")

# DIN99
ax5 = fig.add_subplot(235)
ax5.pcolormesh(a_din99, b_din99, rgb_grid, shading="gouraud", alpha=0.3)
ax5.plot(
    np.append(a_din99[-1, :], a_din99[-1, 0]),
    np.append(b_din99[-1, :], b_din99[-1, 0]),
    color="gray",
    linestyle="--",
    label="sRGB Gamut Boundary",
)
ax5.set_xlabel("a_99")
ax5.set_ylabel("b_99")
ax5.set_title(f"CIEDE2000 (ΔE={DELTA_EQS}) in DIN99")
ax5.grid(True, alpha=0.3)
ax5.legend(loc="upper right")
ax5.set_aspect("equal", adjustable="datalim")

# IPT
ax6 = fig.add_subplot(236)
ax6.pcolormesh(P_ipt, T_ipt, rgb_grid, shading="gouraud", alpha=0.3)
ax6.plot(
    np.append(P_ipt[-1, :], P_ipt[-1, 0]),
    np.append(T_ipt[-1, :], T_ipt[-1, 0]),
    color="gray",
    linestyle="--",
    label="sRGB Gamut Boundary",
)
ax6.set_xlabel("P")
ax6.set_ylabel("T")
ax6.set_title(f"CIEDE2000 (ΔE={DELTA_EQS}) in IPT")
ax6.grid(True, alpha=0.3)
ax6.legend(loc="upper right")
ax6.set_aspect("equal", adjustable="datalim")

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
        x_prime,
        y_prime,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    c_x, c_y, c_z = center_xyz[0, 0]
    c_xyz_den = c_x + c_y + c_z
    c_x_coord = c_x / max(c_xyz_den, 1e-10)
    c_y_coord = c_y / max(c_xyz_den, 1e-10)
    ax2.plot(c_x_coord, c_y_coord, "ko", markersize=3)

    ax3.contour(
        u_prime,
        v_prime,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    c_den = c_x + 15 * c_y + 3 * c_z
    c_u = (4 * c_x) / max(c_den, 1e-10)
    c_v = (9 * c_y) / max(c_den, 1e-10)
    ax3.plot(c_u, c_v, "ko", markersize=3)

    # CAM16-UCS
    c_cam16 = colour.XYZ_to_CAM16UCS(center_xyz)
    c_a_cam16 = c_cam16[0, 0, 1]
    c_b_cam16 = c_cam16[0, 0, 2]
    ax4.contour(
        a_cam16,
        b_cam16,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    ax4.plot(c_a_cam16, c_b_cam16, "ko", markersize=3)

    # DIN99
    c_din99 = colour.XYZ_to_DIN99(center_xyz)
    c_a_din99 = c_din99[0, 0, 1]
    c_b_din99 = c_din99[0, 0, 2]
    ax5.contour(
        a_din99,
        b_din99,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    ax5.plot(c_a_din99, c_b_din99, "ko", markersize=3)

    # IPT
    c_ipt = colour.XYZ_to_IPT(center_xyz)
    c_P_ipt = c_ipt[0, 0, 1]
    c_T_ipt = c_ipt[0, 0, 2]
    ax6.contour(
        P_ipt,
        T_ipt,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1,
    )
    ax6.plot(c_P_ipt, c_T_ipt, "ko", markersize=3)

plt.tight_layout()
plt.show()
