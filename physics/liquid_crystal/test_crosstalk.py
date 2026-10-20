"""Tests for the 2D periodic grating and spatial crosstalk model."""

from dataclasses import replace

import numpy as np
import pytest

from liquid_crystal import crosstalk as ct
from liquid_crystal import frank_model as fm

TEST_CELL = fm.CellParams(
    thickness=2.5e-6,
    pretilt_bottom=np.radians(2.0),
    pretilt_top=np.radians(2.0),
    twist=0.0,
    n_nodes=15,
)
TEST_OPTICS = fm.OpticsParams(
    wavelength=550e-9,
    polarizer_angle=np.pi / 4,
    mode="reflective",
)


def test_electrostatics_no_crosstalk():
    grating = ct.GratingParams(crosstalk=False, n_x=16)
    _, v_target = ct.generate_target_voltage(grating)
    pot, ex, ez = ct.electrostatic_field(v_target, TEST_CELL, grating)

    assert np.allclose(ex, 0.0)
    expected_ez = np.tile(
        v_target[:, None] / TEST_CELL.thickness, (1, TEST_CELL.n_nodes)
    )
    assert np.allclose(ez, expected_ez)
    assert np.allclose(pot[:, 0], v_target)
    assert np.allclose(pot[:, -1], 0.0)


def test_electrostatics_with_crosstalk_boundary_conditions():
    grating = ct.GratingParams(crosstalk=True, n_x=32, profile_type="blazed")
    _, v_target = ct.generate_target_voltage(grating)
    pot, ex, ez = ct.electrostatic_field(v_target, TEST_CELL, grating)

    # Substrate at z=0 must match target voltage; top at z=d must be zero ground
    assert np.allclose(pot[:, 0], v_target, atol=1e-10)
    assert np.allclose(pot[:, -1], 0.0, atol=1e-10)
    # Lateral fringing field Ex must be non-zero
    assert np.max(np.abs(ex)) > 1e4


def test_grating_result_shapes():
    grating = ct.GratingParams(n_x=16, max_steps=50)
    res = ct.solve_grating(TEST_CELL, TEST_OPTICS, grating)

    assert res.theta.shape == (16, TEST_CELL.n_nodes)
    assert res.phi.shape == (16, TEST_CELL.n_nodes)
    assert res.optical_phase.shape == (16,)
    assert res.optical_transmission.shape == (16,)
    assert len(res.orders) == len(res.efficiencies)
    assert np.all(res.efficiencies >= 0.0)
    assert np.sum(res.efficiencies) <= 1.0001


def test_sinusoidal_grating_harmonic_symmetry():
    grating = ct.GratingParams(
        n_x=32,
        profile_type="sinusoidal",
        crosstalk=True,
        max_steps=100,
    )
    res = ct.solve_grating(TEST_CELL, TEST_OPTICS, grating)

    # For a symmetric cosine modulation, positive and negative orders have equal efficiencies
    order_dict = dict(zip(res.orders, res.efficiencies))
    for n in (1, 2, 3):
        if n in order_dict and -n in order_dict:
            assert order_dict[n] == pytest.approx(order_dict[-n], rel=1e-2, abs=1e-4)


def test_crosstalk_smooths_blazed_flyback():
    grating_ideal = ct.GratingParams(
        n_x=32,
        profile_type="blazed",
        crosstalk=False,
    )
    grating_crosstalk = ct.GratingParams(
        n_x=32,
        profile_type="blazed",
        crosstalk=True,
        max_steps=200,
    )

    res_ideal = ct.solve_grating(TEST_CELL, TEST_OPTICS, grating_ideal)
    res_crosstalk = ct.solve_grating(TEST_CELL, TEST_OPTICS, grating_crosstalk)

    # Periodic lateral gradient max |dtheta| across the wrap-around boundary
    # should be significantly smaller with crosstalk due to elastic smoothing
    # of the sharp flyback jump
    grad_ideal = np.max(np.abs(np.roll(res_ideal.theta, -1, axis=0) - res_ideal.theta))
    grad_crosstalk = np.max(
        np.abs(np.roll(res_crosstalk.theta, -1, axis=0) - res_crosstalk.theta)
    )
    assert grad_crosstalk < grad_ideal
