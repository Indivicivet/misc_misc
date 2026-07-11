import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb, rgb2lab, deltaE_ciede2000

# Define the Hue-Saturation space resolution
TIMES_PRIS = 6
DELTA_EQS = [1, 2]
grid_size = 500
h_vals = np.linspace(0, 1, grid_size)
s_vals = np.linspace(0, 1, grid_size)
H, S = np.meshgrid(h_vals, s_vals)
V = np.ones_like(H)

# Convert the entire HS grid to CIELAB
hsv_grid = np.dstack((H, S, V))
rgb_grid = hsv2rgb(hsv_grid)
lab_grid = rgb2lab(rgb_grid)

# Define center points for the ellipses (Hue, Saturation, Value)
centers_hsv = [
    (t, sat, 1)
    for t in np.linspace(0, 1, 3 * TIMES_PRIS + 1)[:-1]
    for sat in np.linspace(0, 1, 7) ** 1.5
]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

# Background color mapping for context
ax.pcolormesh(H * 2 * np.pi, S, rgb_grid, shading="gouraud", alpha=0.3)

for ch, cs, cv in centers_hsv:
    # Convert center point to LAB
    center_rgb = hsv2rgb(np.array([[[ch, cs, cv]]]))
    center_lab = rgb2lab(center_rgb)

    # Calculate CIEDE2000 distance from center to all points in the grid
    delta_e_grid = deltaE_ciede2000(center_lab, lab_grid)
    ax.contour(
        H * 2 * np.pi,
        S,
        delta_e_grid,
        levels=DELTA_EQS,
        colors=["navy", "black"],
        linewidths=1
    )

    # Mark the center point
    ax.plot(ch * 2 * np.pi, cs, "ko", markersize=3)

ax.set_yticks([])
ax.set_xticks(np.linspace(0, 2 * np.pi, 6, endpoint=False))
ax.set_xticklabels(["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"])
ax.set_title(f"CIEDE2000 Contours (ΔE={DELTA_EQS}) in sRGB Hue-Saturation Space (V=1)")

plt.show()
