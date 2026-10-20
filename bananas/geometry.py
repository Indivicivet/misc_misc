"""Botanical geometry engine for a full 360-degree, two-layer banana ring.

Supports dynamic inner and outer finger counts.
Ground state is 3-sided (tricarpellary symmetry). Contact pressure produces
4- and 5-sided profiles.
"""

from dataclasses import dataclass, field
import math
from typing import List, Tuple
import numpy as np

from contact_engine import relax_packing_energy, compute_contact_shape


@dataclass
class BananaFinger:
    """Represents a single banana fruit's transverse cross-section."""

    id: int
    tier: str  # "inner" or "outer"
    center: np.ndarray  # [x, y] in mm
    radius: float
    rotation: float
    facet_count: int
    locule_angles: np.ndarray

    peel_outer_pts: np.ndarray = field(default=None)
    peel_inner_pts: np.ndarray = field(default=None)
    rib_vertex_indices: List[int] = field(default_factory=list)
    curvatures: np.ndarray = field(default=None)
    vascular_bundles: np.ndarray = field(default=None)
    ovule_positions: np.ndarray = field(default=None)
    ovule_radii: np.ndarray = field(default=None)


@dataclass
class CentralStem:
    """Central rachis / stalk around which the bananas are whorled."""

    center: np.ndarray
    radius: float
    boundary_pts: np.ndarray
    vascular_bundles: np.ndarray


@dataclass
class FullRingGeometry:
    """Complete 360-degree two-layer banana ring geometry."""

    stem: CentralStem
    fingers: List[BananaFinger]
    n_inner: int
    n_outer: int
    compression: float
    seed_morphology: float


def generate_full_ring_geometry(
    n_inner: int = 8,
    n_outer: int = 12,
    compression: float = 0.90,
    seed_morphology: float = 0.0,
    random_seed: int = 101,
) -> FullRingGeometry:
    """Generate energy-minimized, tightly packed 360-degree banana ring geometry.

    Dynamically sizes radii for user-specified n_inner and n_outer finger counts.
    """
    rng = np.random.default_rng(random_seed)

    target_r = 22.0

    # 1. Central Rachis (Stem)
    # Stem radius scales with inner banana count so they fit naturally around it
    stem_center = np.array([0.0, 0.0])
    stem_r = max(28.0, 38.0 * (n_inner / 8.0))
    t_stem = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    stem_pts = stem_center + np.stack(
        [
            stem_r * (1.0 + 0.015 * np.sin(4 * t_stem)) * np.cos(t_stem),
            stem_r * (1.0 + 0.015 * np.sin(4 * t_stem)) * np.sin(t_stem),
        ],
        axis=1,
    )

    n_stem_vessels = int(80 + 15 * (n_inner / 8.0))
    v_r = np.sqrt(rng.uniform(0.06, 0.88, n_stem_vessels)) * stem_r
    v_th = rng.uniform(0, 2 * np.pi, n_stem_vessels)
    stem_vessels = np.stack(
        [v_r * np.cos(v_th), v_r * np.sin(v_th)],
        axis=1,
    )
    stem = CentralStem(
        center=stem_center,
        radius=stem_r,
        boundary_pts=stem_pts,
        vascular_bundles=stem_vessels,
    )

    # 2. Dynamic Initial Layout Radii
    # Inner ring radius
    r_inner_init = stem_r + target_r - 6.0 * compression
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    inner_centers_init = np.stack(
        [r_inner_init * np.cos(inner_angles), r_inner_init * np.sin(inner_angles)],
        axis=1,
    )

    # Outer ring radius
    r_outer_init = r_inner_init + 30.5
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + (
        np.pi / n_outer
    )
    outer_centers_init = np.stack(
        [r_outer_init * np.cos(outer_angles), r_outer_init * np.sin(outer_angles)],
        axis=1,
    )

    # 3. Energy minimization / jiggle relaxation
    inner_centers, outer_centers = relax_packing_energy(
        inner_centers=inner_centers_init,
        outer_centers=outer_centers_init,
        stem_center=stem_center,
        stem_radius=stem_r,
        banana_radius=target_r,
        n_iterations=280,
        compression=compression,
        seed=random_seed,
    )

    fingers: List[BananaFinger] = []

    # Process Inner Ring Bananas
    for i in range(n_inner):
        c_i = inner_centers[i]
        f_seed = int(rng.integers(1000, 99999))

        neighbors = [
            inner_centers[(i - 1) % n_inner],
            inner_centers[(i + 1) % n_inner],
        ]
        for o_c in outer_centers:
            if np.linalg.norm(c_i - o_c) < target_r * 2.15:
                neighbors.append(o_c)

        rot = math.atan2(c_i[1], c_i[0])
        # 3 Carpel locules oriented radially with subtle biological drift
        loc_rot = rot + rng.uniform(-0.06, 0.06)
        loc_angles = np.array(
            [loc_rot, loc_rot + 2 * np.pi / 3, loc_rot + 4 * np.pi / 3]
        )

        outer_pts, rib_idx, curv, n_facets = compute_contact_shape(
            center=c_i,
            target_radius=target_r,
            stem_center=stem_center,
            stem_radius=stem_r,
            neighbor_centers=neighbors,
            locule_orientation=loc_rot,
            compression=compression,
            corner_fillet=2.0,
            seed=f_seed,
        )

        peel_thick = 3.6
        inner_pts, _, _, _ = compute_contact_shape(
            center=c_i,
            target_radius=target_r - peel_thick,
            stem_center=stem_center,
            stem_radius=stem_r + peel_thick * 0.4,
            neighbor_centers=neighbors,
            locule_orientation=loc_rot,
            compression=compression * 0.78,
            corner_fillet=2.5,
            seed=f_seed,
        )

        n_bundles = 32
        b_idx = np.linspace(0, len(inner_pts) - 1, n_bundles, dtype=int)
        v_bundles = inner_pts[b_idx] + (outer_pts[b_idx] - inner_pts[b_idx]) * 0.22

        n_seeds = 5
        seed_pos_list = []
        seed_r_list = []
        for l_ang in loc_angles:
            for s_i in range(n_seeds):
                dist = 0.8 + (s_i + 0.6) * (target_r * 0.23 / n_seeds)
                lat = (-1 if s_i % 2 == 0 else 1) * 0.48 * (1.0 + seed_morphology * 1.4)
                sx = (
                    c_i[0]
                    + dist * np.cos(l_ang)
                    + lat * np.sin(l_ang)
                    + rng.uniform(-0.15, 0.15)
                )
                sy = (
                    c_i[1]
                    + dist * np.sin(l_ang)
                    - lat * np.cos(l_ang)
                    + rng.uniform(-0.15, 0.15)
                )
                seed_pos_list.append([sx, sy])
                s_radius = 0.35 + seed_morphology * 1.85 + rng.uniform(-0.06, 0.06)
                seed_r_list.append(s_radius)

        finger = BananaFinger(
            id=i + 1,
            tier="inner",
            center=c_i,
            radius=target_r,
            rotation=rot,
            facet_count=n_facets,
            locule_angles=loc_angles,
            peel_outer_pts=outer_pts,
            peel_inner_pts=inner_pts,
            rib_vertex_indices=rib_idx,
            curvatures=curv,
            vascular_bundles=v_bundles,
            ovule_positions=np.array(seed_pos_list),
            ovule_radii=np.array(seed_r_list),
        )
        fingers.append(finger)

    # Process Outer Ring Bananas
    for j in range(n_outer):
        c_j = outer_centers[j]
        f_seed = int(rng.integers(1000, 99999))

        neighbors = [
            outer_centers[(j - 1) % n_outer],
            outer_centers[(j + 1) % n_outer],
        ]
        for i_c in inner_centers:
            if np.linalg.norm(c_j - i_c) < target_r * 2.15:
                neighbors.append(i_c)

        rot = math.atan2(c_j[1], c_j[0])
        loc_rot = rot + rng.uniform(-0.06, 0.06)
        loc_angles = np.array(
            [loc_rot, loc_rot + 2 * np.pi / 3, loc_rot + 4 * np.pi / 3]
        )

        outer_pts, rib_idx, curv, n_facets = compute_contact_shape(
            center=c_j,
            target_radius=target_r,
            stem_center=stem_center,
            stem_radius=stem_r,
            neighbor_centers=neighbors,
            locule_orientation=loc_rot,
            compression=compression,
            corner_fillet=2.0,
            seed=f_seed,
        )

        peel_thick = 3.6
        inner_pts, _, _, _ = compute_contact_shape(
            center=c_j,
            target_radius=target_r - peel_thick,
            stem_center=stem_center,
            stem_radius=stem_r + peel_thick * 0.4,
            neighbor_centers=neighbors,
            locule_orientation=loc_rot,
            compression=compression * 0.78,
            corner_fillet=2.5,
            seed=f_seed,
        )

        n_bundles = 32
        b_idx = np.linspace(0, len(inner_pts) - 1, n_bundles, dtype=int)
        v_bundles = inner_pts[b_idx] + (outer_pts[b_idx] - inner_pts[b_idx]) * 0.22

        n_seeds = 5
        seed_pos_list = []
        seed_r_list = []
        for l_ang in loc_angles:
            for s_i in range(n_seeds):
                dist = 0.8 + (s_i + 0.6) * (target_r * 0.23 / n_seeds)
                lat = (-1 if s_i % 2 == 0 else 1) * 0.48 * (1.0 + seed_morphology * 1.4)
                sx = (
                    c_j[0]
                    + dist * np.cos(l_ang)
                    + lat * np.sin(l_ang)
                    + rng.uniform(-0.15, 0.15)
                )
                sy = (
                    c_j[1]
                    + dist * np.sin(l_ang)
                    - lat * np.cos(l_ang)
                    + rng.uniform(-0.15, 0.15)
                )
                seed_pos_list.append([sx, sy])
                s_radius = 0.35 + seed_morphology * 1.85 + rng.uniform(-0.06, 0.06)
                seed_r_list.append(s_radius)

        finger = BananaFinger(
            id=n_inner + j + 1,
            tier="outer",
            center=c_j,
            radius=target_r,
            rotation=rot,
            facet_count=n_facets,
            locule_angles=loc_angles,
            peel_outer_pts=outer_pts,
            peel_inner_pts=inner_pts,
            rib_vertex_indices=rib_idx,
            curvatures=curv,
            vascular_bundles=v_bundles,
            ovule_positions=np.array(seed_pos_list),
            ovule_radii=np.array(seed_r_list),
        )
        fingers.append(finger)

    return FullRingGeometry(
        stem=stem,
        fingers=fingers,
        n_inner=n_inner,
        n_outer=n_outer,
        compression=compression,
        seed_morphology=seed_morphology,
    )
