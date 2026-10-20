"""Checks of the Frank model against analytic results."""

from dataclasses import replace

import numpy as np
import pytest

from liquid_crystal import frank_model as fm

PLANAR = fm.CellParams(pretilt_bottom=0.0, pretilt_top=0.0, twist=0.0)
TN = fm.CellParams(pretilt_bottom=0.0, pretilt_top=0.0, twist=np.pi / 2)
TN_CHIRAL = replace(TN, pitch=20e-6)
VAN = fm.CellParams(
    k11=16.7e-12,
    k33=18.1e-12,
    eps_perp=7.8,
    delta_eps=-4.2,
    pretilt_bottom=np.pi / 2,
    pretilt_top=np.pi / 2,
    twist=0.0,
)


def test_gradient_matches_finite_differences():
    p = replace(TN_CHIRAL, n_nodes=15)
    rng = np.random.default_rng(0)
    theta = rng.uniform(-1.0, 1.0, p.n_nodes)
    phi = rng.uniform(-2.0, 2.0, p.n_nodes)
    voltage = 3.0
    _, grad_theta, grad_phi = fm.free_energy_and_gradient(theta, phi, p, voltage)
    step = 1e-6
    for i in range(p.n_nodes):
        for values, grad in ((theta, grad_theta), (phi, grad_phi)):
            plus = values.copy()
            minus = values.copy()
            plus[i] += step
            minus[i] -= step
            args_plus = (plus, phi) if values is theta else (theta, plus)
            args_minus = (minus, phi) if values is theta else (theta, minus)
            numeric = (
                fm.free_energy_and_gradient(*args_plus, p, voltage)[0]
                - fm.free_energy_and_gradient(*args_minus, p, voltage)[0]
            ) / (2 * step)
            assert numeric == pytest.approx(grad[i], rel=1e-5, abs=1e-12)


def _max_tilt_deviation(p, voltage):
    theta, phi = fm.equilibrium(*fm.initial_state(p), p, voltage)
    reference = 0.5 * (p.pretilt_bottom + p.pretilt_top)
    return np.degrees(np.max(np.abs(theta - reference)))


@pytest.mark.parametrize(
    "p", [PLANAR, TN, TN_CHIRAL, VAN], ids=["planar", "tn", "tn_chiral", "van"]
)
def test_freedericksz_threshold(p):
    v_th = fm.freedericksz_threshold(p)
    assert _max_tilt_deviation(p, 0.97 * v_th) < 0.5
    assert _max_tilt_deviation(p, 1.03 * v_th) > 3.0


def test_tn_mauguin_off_state_transmits():
    p = replace(TN, pretilt_bottom=np.radians(2), pretilt_top=np.radians(2))
    theta, phi = fm.equilibrium(*fm.initial_state(p), p, 0.0)
    assert fm.optical_output(theta, phi, p, fm.OpticsParams()) > 0.99


def test_planar_retarder_matches_analytic():
    p = PLANAR
    optics = fm.OpticsParams(polarizer_angle=np.pi / 4)
    theta, phi = fm.initial_state(p, perturbation=0.0)
    expected = np.sin(np.pi * p.thickness * optics.delta_n / optics.wavelength) ** 2
    assert fm.optical_output(theta, phi, p, optics) == pytest.approx(expected)


def test_van_reflective_off_state_is_dark():
    optics = fm.OpticsParams(
        polarizer_angle=np.pi / 4, delta_n=0.083, mode="reflective"
    )
    theta, phi = fm.initial_state(VAN, perturbation=0.0)
    assert fm.optical_output(theta, phi, VAN, optics) == pytest.approx(0.0, abs=1e-12)


def test_dynamics_reaches_equilibrium():
    p = replace(TN, pretilt_bottom=np.radians(2), pretilt_top=np.radians(2))
    start = fm.initial_state(p)
    theta_eq, _ = fm.equilibrium(*start, p, 3.0)
    theta_dyn, _, _, _ = fm.relax_dynamics(*start, p, 3.0, 0.3, max_steps=10**6)
    assert np.max(np.abs(theta_dyn - theta_eq)) < np.radians(0.2)
