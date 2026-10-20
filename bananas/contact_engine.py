"""Physics-based contact packing and deformable shape simulation for a 360-degree banana ring."""

import math
from typing import List, Tuple
import numpy as np


def relax_packing_energy(
    inner_centers: np.ndarray,
    outer_centers: np.ndarray,
    stem_center: np.ndarray,
    stem_radius: float,
    banana_radius: float,
    n_iterations: int = 300,
    compression: float = 0.92,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Energy minimization with simulated jiggle to densely pack the bananas.

    Simulates physical inward centripetal pressure and repulsive elastic contact:
    - Inner bananas are pressed against the stem and compressed laterally against each other.
    - Outer bananas are pulled radially inward, settling into the interstitial valleys
      between inner bananas and compressing laterally against adjacent outer bananas.
    """
    rng = np.random.default_rng(seed)
    n_in = len(inner_centers)
    n_out = len(outer_centers)

    pos_in = np.copy(inner_centers)
    pos_out = np.copy(outer_centers)

    stem_overlap = 7.5 * compression
    inter_banana_overlap = 6.2 * compression

    target_d_stem = stem_radius + banana_radius - stem_overlap
    target_d_pair = 2.0 * banana_radius - inter_banana_overlap

    dt = 0.075
    damping = 0.84
    vel_in = np.zeros_like(pos_in)
    vel_out = np.zeros_like(pos_out)

    for step in range(n_iterations):
        temp = max(0.0, 1.0 - step / (n_iterations * 0.70))
        jiggle_mag = 0.40 * temp

        forces_in = np.zeros_like(pos_in)
        forces_out = np.zeros_like(pos_out)

        # 1. Inner bananas: Stem contact force & inward pressure
        for i in range(n_in):
            v_stem = pos_in[i] - stem_center
            d_stem = np.linalg.norm(v_stem)
            n_stem = v_stem / max(d_stem, 1e-6)

            diff = d_stem - target_d_stem
            forces_in[i] -= 22.0 * diff * n_stem

            tangent = np.array([-n_stem[1], n_stem[0]])
            forces_in[i] += rng.uniform(-jiggle_mag, jiggle_mag) * tangent

        # 2. Inner-Inner lateral contact forces
        for i in range(n_in):
            for j in range(i + 1, n_in):
                v_ij = pos_in[j] - pos_in[i]
                d_ij = np.linalg.norm(v_ij)
                if d_ij < target_d_pair * 1.30:
                    n_ij = v_ij / max(d_ij, 1e-6)
                    f_repel = 26.0 * (target_d_pair - d_ij) * n_ij
                    forces_in[i] -= f_repel
                    forces_in[j] += f_repel

        # 3. Outer bananas: Inward radial pulling force (deep valley packing)
        for j in range(n_out):
            v_stem = pos_out[j] - stem_center
            d_stem = np.linalg.norm(v_stem)
            n_stem = v_stem / max(d_stem, 1e-6)

            forces_out[j] -= 32.0 * n_stem

            tangent = np.array([-n_stem[1], n_stem[0]])
            forces_out[j] += rng.uniform(-jiggle_mag * 1.6, jiggle_mag * 1.6) * tangent

        # 4. Outer-Inner contact forces
        for j in range(n_out):
            for i in range(n_in):
                v_oi = pos_out[j] - pos_in[i]
                d_oi = np.linalg.norm(v_oi)
                if d_oi < target_d_pair:
                    n_oi = v_oi / max(d_oi, 1e-6)
                    f_repel = 34.0 * (target_d_pair - d_oi) * n_oi
                    forces_out[j] += f_repel
                    forces_in[i] -= f_repel * 0.35

        # 5. Outer-Outer lateral contact forces
        for j in range(n_out):
            for k in range(j + 1, n_out):
                v_jk = pos_out[k] - pos_out[j]
                d_jk = np.linalg.norm(v_jk)
                if d_jk < target_d_pair * 1.30:
                    n_jk = v_jk / max(d_jk, 1e-6)
                    f_repel = 28.0 * (target_d_pair - d_jk) * n_jk
                    forces_out[j] -= f_repel
                    forces_out[k] += f_repel

        vel_in = vel_in * damping + forces_in * dt
        vel_out = vel_out * damping + forces_out * dt

        vel_in = np.clip(vel_in, -3.0, 3.0)
        vel_out = np.clip(vel_out, -3.0, 3.0)

        pos_in += vel_in * dt
        pos_out += vel_out * dt

    return pos_in, pos_out


def compute_contact_shape(
    center: np.ndarray,
    target_radius: float,
    stem_center: np.ndarray,
    stem_radius: float,
    neighbor_centers: List[np.ndarray],
    compression: float = 0.92,
    corner_fillet: float = 2.2,
    n_samples: int = 360,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int], np.ndarray, int]:
    """Compute the true contact-flattened cross-section of a banana pressed into neighbors and stem."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    ray_dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    r_base = target_radius * (
        1.0 + 0.015 * np.cos(3 * theta + rng.uniform(0, 2 * np.pi))
    )

    planes = []

    # 1. Stem contact plane
    v_stem = stem_center - center
    d_stem = np.linalg.norm(v_stem)
    clearance_stem = d_stem - stem_radius
    if clearance_stem < target_radius:
        n_stem = v_stem / d_stem
        p_dist = clearance_stem + (target_radius - clearance_stem) * (1.0 - compression)
        planes.append((n_stem, p_dist))

    # 2. Neighbor contact planes
    for n_c in neighbor_centers:
        v_n = n_c - center
        d_n = np.linalg.norm(v_n)
        if d_n < target_radius * 2.05:
            n_n = v_n / d_n
            p_dist = (d_n / 2.0) + (target_radius - d_n / 2.0) * (1.0 - compression)
            planes.append((n_n, p_dist))

    r_clipped = np.copy(r_base)
    for n_vec, p_dist in planes:
        dots = np.dot(ray_dirs, n_vec)
        active = dots > 0.02
        r_plane = p_dist / np.maximum(dots[active], 0.02)
        r_clipped[active] = np.minimum(r_clipped[active], r_plane)

    # Gaussian circular smoothing for corner filleting
    kernel_size = 23
    sigma = corner_fillet * (1.0 - 0.40 * compression)
    x_k = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel = np.exp(-0.5 * (x_k / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.concatenate(
        [r_clipped[-kernel_size:], r_clipped, r_clipped[:kernel_size]]
    )
    r_smooth = np.convolve(padded, kernel, mode="same")[kernel_size:-kernel_size]

    points = center + np.stack(
        [r_smooth * np.cos(theta), r_smooth * np.sin(theta)], axis=1
    )

    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = dx * ddy - dy * ddx
    den = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.abs(num / np.maximum(den, 1e-6))

    # Identify rib vertices (curvature peaks)
    rib_indices = []
    n_pts = len(points)
    curv_threshold = np.mean(curvature) + 0.28 * np.std(curvature)
    min_spacing = int(n_pts * 0.11)

    candidates = []
    for i in range(n_pts):
        prev_i = (i - 1) % n_pts
        next_i = (i + 1) % n_pts
        if curvature[i] > curvature[prev_i] and curvature[i] > curvature[next_i]:
            if curvature[i] > curv_threshold:
                candidates.append((curvature[i], i))

    candidates.sort(key=lambda c: c[0], reverse=True)
    selected = []
    for c_val, idx in candidates:
        if not any(
            min(abs(idx - s), n_pts - abs(idx - s)) < min_spacing for s in selected
        ):
            selected.append(idx)

    selected.sort()
    facet_count = len(selected)
    facet_count = max(3, min(5, facet_count if facet_count >= 3 else 4))

    return points, selected, curvature, facet_count
