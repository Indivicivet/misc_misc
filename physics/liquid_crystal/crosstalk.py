"""2D periodic grating and spatial crosstalk model for liquid-crystal cells.

Models the lateral spatial crosstalk between adjacent regions/pixels in a nematic LC
layer (such as an SLM or LCoS microdisplay) across one spatial period Lambda along x,
with thickness d along z.

The model captures:
1. Electrostatic fringing fields (decay of higher spatial harmonics into the cell bulk).
2. Frank continuum elastic coupling (lateral splay/bend/twist smoothing).
3. Generation of spatial optical harmonics: e^{i k x} -> sum_n c_n e^{i n k x}
   (harmonic spillover, flyback blunting, and diffraction efficiencies).

Enabled by default via `GratingParams(crosstalk=True)`. Setting `crosstalk=False`
recovers independent 1D column behaviour (ideal uncoupled pixels).
"""

from dataclasses import dataclass

import numpy as np

from liquid_crystal import frank_model as fm

PROFILE_TYPES = ("blazed", "sinusoidal", "binary")


@dataclass(frozen=True)
class GratingParams:
    """Parameters for a 1D lateral periodic grating across x in meters."""

    period: float = 10e-6
    n_x: int = 32
    profile_type: str = "blazed"
    v_min: float = 0.5
    v_max: float = 4.5
    crosstalk: bool = True
    include_fringing_field: bool = True
    include_elastic_crosstalk: bool = True
    k_lateral: float | None = None
    max_steps: int = 1200


@dataclass(frozen=True)
class GratingResult:
    """Simulation results across one lateral grating period Lambda."""

    x: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    v_target: np.ndarray
    potential: np.ndarray
    e_x: np.ndarray
    e_z: np.ndarray
    optical_phase: np.ndarray
    optical_transmission: np.ndarray
    orders: np.ndarray
    harmonics_c: np.ndarray
    efficiencies: np.ndarray


def generate_target_voltage(grating: GratingParams) -> tuple[np.ndarray, np.ndarray]:
    """Return lateral coordinates x and target electrode voltage V_target(x)."""
    x = np.linspace(0.0, grating.period, grating.n_x, endpoint=False)
    span = grating.v_max - grating.v_min
    if grating.profile_type == "blazed":
        # Sawtooth blazed profile: linear ramp from v_min to v_max
        v = grating.v_min + span * (x / grating.period)
    elif grating.profile_type == "sinusoidal":
        v = grating.v_min + 0.5 * span * (1.0 + np.cos(2 * np.pi * x / grating.period))
    elif grating.profile_type == "binary":
        v = np.where(x < 0.5 * grating.period, grating.v_max, grating.v_min)
    else:
        raise ValueError(f"Unknown profile_type: {grating.profile_type}")
    return x, v


def electrostatic_field(
    v_target: np.ndarray, cell: fm.CellParams, grating: GratingParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (potential V(x,z), E_x(x,z), E_z(x,z)) across one spatial period.

    For crosstalk=True and include_fringing_field=True, solves Laplace's equation
    spectrally in x with boundary conditions V(x, 0) = V_target(x) and V(x, d) = 0.
    Higher spatial harmonics decay as sinh(q_m (d - z)) / sinh(q_m d).
    """
    n_x = grating.n_x
    z = cell.z
    dx = grating.period / n_x

    if not (grating.crosstalk and grating.include_fringing_field):
        # 1D parallel-plate approximation at each column (no fringing)
        frac = 1.0 - z / cell.thickness
        potential = v_target[:, None] * frac[None, :]
        e_x = np.zeros((n_x, cell.n_nodes))
        e_z = np.tile(v_target[:, None] / cell.thickness, (1, cell.n_nodes))
        return potential, e_x, e_z

    v_fft = np.fft.rfft(v_target)
    freqs = np.fft.rfftfreq(n_x, d=dx)
    q = 2 * np.pi * freqs

    potential = np.zeros((n_x, cell.n_nodes))
    e_x = np.zeros((n_x, cell.n_nodes))
    e_z = np.zeros((n_x, cell.n_nodes))

    for j, zj in enumerate(z):
        mode_z = np.zeros_like(v_fft, dtype=complex)
        mode_z_deriv = np.zeros_like(v_fft, dtype=complex)

        # m = 0 fundamental DC mode
        mode_z[0] = v_fft[0] * (1.0 - zj / cell.thickness)
        mode_z_deriv[0] = -v_fft[0] / cell.thickness

        if len(q) > 1:
            qd = q[1:] * cell.thickness
            qd_rem = q[1:] * (cell.thickness - zj)
            # Clip qd for numerical stability against sinh overflow
            qd_safe = np.clip(qd, 0.0, 70.0)
            qd_rem_safe = np.clip(qd_rem, 0.0, 70.0)
            sinh_qd = np.sinh(qd_safe)
            mode_z[1:] = v_fft[1:] * (np.sinh(qd_rem_safe) / sinh_qd)
            mode_z_deriv[1:] = -v_fft[1:] * q[1:] * (np.cosh(qd_rem_safe) / sinh_qd)

        potential[:, j] = np.fft.irfft(mode_z, n=n_x)
        e_x[:, j] = np.fft.irfft(-1j * q * mode_z, n=n_x)
        e_z[:, j] = np.fft.irfft(-mode_z_deriv, n=n_x)

    return potential, e_x, e_z


def grating_layer_jones_matrices(
    theta: np.ndarray, phi: np.ndarray, cell: fm.CellParams, optics: fm.OpticsParams
) -> np.ndarray:
    """Compute Jones matrices of each slab for all columns (n_x, n_links, 2, 2)."""
    theta_mid = 0.5 * (theta[:, :-1] + theta[:, 1:])
    phi_mid = 0.5 * (phi[:, :-1] + phi[:, 1:])
    n_e = optics.n_o + optics.delta_n
    n_eff = 1.0 / np.sqrt(
        np.sin(theta_mid) ** 2 / optics.n_o**2 + np.cos(theta_mid) ** 2 / n_e**2
    )
    retardance = 2 * np.pi * (n_eff - optics.n_o) * cell.h / optics.wavelength
    slow = np.exp(-0.5j * retardance)
    fast = np.exp(0.5j * retardance)
    c = np.cos(phi_mid)
    s = np.sin(phi_mid)
    off_diagonal = (slow - fast) * c * s

    row0 = np.stack([slow * c**2 + fast * s**2, off_diagonal], axis=-1)
    row1 = np.stack([off_diagonal, slow * s**2 + fast * c**2], axis=-1)
    return np.stack([row0, row1], axis=-2)


def grating_optical_fields(
    theta: np.ndarray, phi: np.ndarray, cell: fm.CellParams, optics: fm.OpticsParams
) -> np.ndarray:
    """Return complex output field for all columns (n_x,)."""
    n_x = theta.shape[0]
    m_slabs = grating_layer_jones_matrices(theta, phi, cell, optics)
    if optics.mode == "reflective":
        m_slabs = m_slabs[:, ::-1, :, :]

    jones = np.broadcast_to(np.eye(2, dtype=complex), (n_x, 2, 2)).copy()
    for link in range(m_slabs.shape[1]):
        jones = m_slabs[:, link] @ jones

    if optics.mode == "reflective":
        jones = np.swapaxes(jones, -1, -2) @ jones

    alpha = optics.polarizer_angle
    e_in = np.array([np.cos(alpha), np.sin(alpha)])
    e_out = np.einsum("bij,j->bi", jones, e_in)

    if optics.analyser_angle is not None:
        beta = optics.analyser_angle
        e_analyser = np.array([np.cos(beta), np.sin(beta)])
        return np.einsum("i,bi->b", e_analyser, e_out)

    # In reflective LCoS mode without explicit analyser, measure output along input axis
    return np.einsum("i,bi->b", e_in, e_out)


def compute_diffraction_harmonics(
    optical_field: np.ndarray, max_order: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial Fourier harmonics e^{i n k x} and diffraction efficiencies.

    Returns (orders, complex_harmonics, efficiencies).
    """
    n_x = optical_field.shape[0]
    harmonics_fft = np.fft.fft(optical_field) / n_x
    orders = np.fft.fftfreq(n_x) * n_x

    sort_idx = np.argsort(orders)
    orders = orders[sort_idx]
    harmonics_fft = harmonics_fft[sort_idx]
    efficiencies = np.abs(harmonics_fft) ** 2

    mask = (orders >= -max_order) & (orders <= max_order)
    return orders[mask].astype(int), harmonics_fft[mask], efficiencies[mask]


def solve_grating(
    cell: fm.CellParams,
    optics: fm.OpticsParams,
    grating: GratingParams = GratingParams(),
) -> GratingResult:
    """Solve 2D director profile and optical harmonics for a periodic grating."""
    x, v_target = generate_target_voltage(grating)
    n_x = grating.n_x
    h = cell.h
    dx = grating.period / n_x

    potential, ex, ez = electrostatic_field(v_target, cell, grating)

    # Initialise each column at 1D equilibrium
    theta = np.zeros((n_x, cell.n_nodes))
    phi = np.zeros((n_x, cell.n_nodes))
    for i in range(n_x):
        th_init, ph_init = fm.initial_state(cell)
        theta[i], phi[i] = fm.equilibrium(th_init, ph_init, cell, v_target[i])

    if grating.crosstalk and grating.include_elastic_crosstalk:
        k_lat = (
            grating.k_lateral
            if grating.k_lateral is not None
            else (cell.k11 + cell.k22 + cell.k33) / 3.0
        )
        max_ez = max(float(np.max(np.abs(ez))), 1.0)
        dt = (
            0.35
            * cell.gamma1
            / (
                4 * max(cell.k11, cell.k33) * (1.0 / h**2 + 1.0 / dx**2)
                + 2 * fm.EPS0 * abs(cell.delta_eps) * max_ez**2
            )
        )

        for _ in range(grating.max_steps):
            # Vectorized 1D Frank terms along z across all columns
            theta_mid = 0.5 * (theta[:, :-1] + theta[:, 1:])
            d_theta = np.diff(theta, axis=-1) / h
            d_phi = np.diff(phi, axis=-1) / h
            cos_sq = np.cos(theta_mid) ** 2
            sin_sq = 1.0 - cos_sq
            twist_arg = cos_sq * d_phi - cell.q0
            sin_2t = np.sin(2 * theta_mid)

            df_dtheta_mid = sin_2t * (
                -0.5 * cell.k11 * d_theta**2
                - cell.k22 * twist_arg * d_phi
                + 0.5 * cell.k33 * (d_theta**2 + np.cos(2 * theta_mid) * d_phi**2)
            )
            df_dd_theta = (cell.k11 * cos_sq + cell.k33 * sin_sq) * d_theta
            df_dd_phi = cos_sq * (cell.k22 * twist_arg + cell.k33 * sin_sq * d_phi)

            grad_theta = np.zeros_like(theta)
            grad_theta[:, :-1] += 0.5 * h * df_dtheta_mid - df_dd_theta
            grad_theta[:, 1:] += 0.5 * h * df_dtheta_mid + df_dd_theta

            grad_phi = np.zeros_like(phi)
            grad_phi[:, :-1] -= df_dd_phi
            grad_phi[:, 1:] += df_dd_phi

            dtheta_dt = -grad_theta / (cell.gamma1 * h)
            phi_mob = 1.0 / (np.cos(theta[:, 1:-1]) ** 2 + 1e-4)
            dphi_dt = np.zeros_like(phi)
            dphi_dt[:, 1:-1] = -grad_phi[:, 1:-1] * phi_mob / (cell.gamma1 * h)

            # Lateral Frank elastic coupling along x (periodic)
            theta_left = np.roll(theta, 1, axis=0)
            theta_right = np.roll(theta, -1, axis=0)
            dtheta_dt += (
                (k_lat / cell.gamma1) * (theta_right - 2 * theta + theta_left) / dx**2
            )

            phi_left = np.roll(phi, 1, axis=0)
            phi_right = np.roll(phi, -1, axis=0)
            dphi_dt += (k_lat / cell.gamma1) * (phi_right - 2 * phi + phi_left) / dx**2

            # 2D electric torque: -d(f_diel)/dtheta
            n_dot_e = ex * np.cos(theta) * np.cos(phi) + ez * np.sin(theta)
            d_ndote_dtheta = -ex * np.sin(theta) * np.cos(phi) + ez * np.cos(theta)
            torque_theta = fm.EPS0 * cell.delta_eps * n_dot_e * d_ndote_dtheta
            dtheta_dt += torque_theta / cell.gamma1

            # Impose strong anchoring boundary conditions
            theta[:, 0] = cell.pretilt_bottom
            theta[:, -1] = cell.pretilt_top
            phi[:, 0] = cell.azimuth_bottom
            phi[:, -1] = cell.azimuth_bottom + cell.twist

            theta[:, 1:-1] += dt * dtheta_dt[:, 1:-1]
            phi[:, 1:-1] += dt * dphi_dt[:, 1:-1]

    # Optical outputs and Fourier harmonics
    fields = grating_optical_fields(theta, phi, cell, optics)
    phases = np.unwrap(np.angle(fields))
    transmissions = np.abs(fields) ** 2
    orders, harmonics_c, effs = compute_diffraction_harmonics(fields)

    return GratingResult(
        x=x,
        z=cell.z,
        theta=theta,
        phi=phi,
        v_target=v_target,
        potential=potential,
        e_x=ex,
        e_z=ez,
        optical_phase=phases,
        optical_transmission=transmissions,
        orders=orders,
        harmonics_c=harmonics_c,
        efficiencies=effs,
    )
