"""1D Frank-Oseen continuum model of a nematic liquid-crystal cell.

The cell lies between two substrates at z = 0 and z = d. The director only varies
along z and is parametrised by a tilt angle theta (elevation above the substrate
plane) and an azimuth phi:

    n = (cos(theta) cos(phi), cos(theta) sin(phi), sin(theta))

Substituting this into the Frank distortion free-energy density

    f = K11/2 (div n)^2 + K22/2 (n . curl n + q0)^2 + K33/2 |n x curl n|^2

gives, with ' = d/dz,

    splay = K11/2 cos^2(theta) theta'^2
    twist = K22/2 (cos^2(theta) phi' - q0)^2
    bend  = K33/2 sin^2(theta) (theta'^2 + cos^2(theta) phi'^2)

(sign convention: q0 = 2 pi / pitch > 0 favours phi increasing with z).

The electric field is along z and the cell is driven at fixed (RMS) voltage V. As
D_z is uniform, E_z(z) = V / (eps_zz(z) * int dz / eps_zz) with
eps_zz = eps_perp + delta_eps sin^2(theta). The thermodynamic potential minimised
at fixed voltage is F_elastic - C V^2 / 2, with C = eps0 / int dz / eps_zz.

Anchoring is strong (fixed angles at both substrates). Backflow, flexoelectricity
and ionic effects are ignored.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

EPS0 = 8.8541878128e-12

# Regularises the azimuthal mobility 1/cos^2(theta), which diverges for vertical
# directors where phi is undefined.
_PHI_MOBILITY_REGULARISATION = 1e-4

OPTICS_MODES = ("transmissive", "reflective")


@dataclass(frozen=True)
class CellParams:
    """Material and geometry parameters in SI units (angles in radians)."""

    k11: float = 11.1e-12
    k22: float = 6.5e-12
    k33: float = 17.1e-12
    eps_perp: float = 5.2
    delta_eps: float = 13.8
    gamma1: float = 0.15
    thickness: float = 5e-6
    # 0 means achiral. Positive pitch favours phi increasing with z.
    pitch: float = 0.0
    twist: float = np.pi / 2
    pretilt_bottom: float = np.radians(2.0)
    pretilt_top: float = np.radians(2.0)
    azimuth_bottom: float = 0.0
    n_nodes: int = 41

    @property
    def q0(self) -> float:
        return 0.0 if self.pitch == 0 else 2 * np.pi / self.pitch

    @property
    def h(self) -> float:
        return self.thickness / (self.n_nodes - 1)

    @property
    def z(self) -> np.ndarray:
        return np.linspace(0.0, self.thickness, self.n_nodes)


@dataclass(frozen=True)
class OpticsParams:
    wavelength: float = 550e-9
    n_o: float = 1.52
    delta_n: float = 0.22
    # Input polariser azimuth (radians) in the lab frame.
    polarizer_angle: float = 0.0
    mode: str = "transmissive"
    # If analyser_angle is None, an orthogonal pair (parallel, crossed) is computed.
    analyser_angle: float | None = None


def _trapezoid_weights(n_nodes: int) -> np.ndarray:
    weights = np.ones(n_nodes)
    weights[[0, -1]] = 0.5
    return weights


def apply_boundaries(theta, phi, p: CellParams):
    """Return copies of (theta, phi) with the strong-anchoring values imposed."""
    theta = theta.copy()
    phi = phi.copy()
    theta[[0, -1]] = p.pretilt_bottom, p.pretilt_top
    phi[[0, -1]] = p.azimuth_bottom, p.azimuth_bottom + p.twist
    return theta, phi


def initial_state(p: CellParams, perturbation: float = 1e-3):
    """Linear interpolation between the anchoring angles plus a small tilt bump.

    The bump breaks the up/down symmetry so that zero-pretilt cells can undergo
    the Freedericksz transition instead of sitting on the unstable symmetric state.
    It always points away from the nearest of theta = 0 or theta = 90 deg.
    """
    s = np.linspace(0.0, 1.0, p.n_nodes)
    theta_lin = p.pretilt_bottom + (p.pretilt_top - p.pretilt_bottom) * s
    bump_sign = -1.0 if np.mean(theta_lin) > np.pi / 4 else 1.0
    theta = theta_lin + bump_sign * perturbation * np.sin(np.pi * s)
    phi = p.azimuth_bottom + p.twist * s
    return apply_boundaries(theta, phi, p)


def resample_state(theta, phi, p: CellParams):
    """Interpolate a state onto the node grid of `p` and impose its boundaries."""
    s_old = np.linspace(0.0, 1.0, theta.size)
    s_new = np.linspace(0.0, 1.0, p.n_nodes)
    return apply_boundaries(
        np.interp(s_new, s_old, theta), np.interp(s_new, s_old, phi), p
    )


def field_profile(theta, p: CellParams, voltage: float):
    """Return (E_z at each node, int dz / eps_zz)."""
    eps_zz = p.eps_perp + p.delta_eps * np.sin(theta) ** 2
    inverse_capacitance_integral = p.h * np.sum(_trapezoid_weights(theta.size) / eps_zz)
    return voltage / (eps_zz * inverse_capacitance_integral), (
        inverse_capacitance_integral
    )


def _link_terms(theta, phi, p: CellParams):
    """Midpoint angle, derivatives and trig factors on each link between nodes."""
    theta_mid = 0.5 * (theta[:-1] + theta[1:])
    d_theta = np.diff(theta) / p.h
    d_phi = np.diff(phi) / p.h
    cos_sq = np.cos(theta_mid) ** 2
    return theta_mid, d_theta, d_phi, cos_sq, 1.0 - cos_sq, cos_sq * d_phi - p.q0


def energy_densities(theta, phi, p: CellParams, voltage: float) -> dict:
    """Frank energy densities (J/m^3) on link midpoints and field terms on nodes."""
    _, d_theta, d_phi, cos_sq, sin_sq, twist_arg = _link_terms(theta, phi, p)
    e_z, _ = field_profile(theta, p, voltage)
    z = p.z
    return {
        "z_links": 0.5 * (z[:-1] + z[1:]),
        "splay": 0.5 * p.k11 * cos_sq * d_theta**2,
        "twist": 0.5 * p.k22 * twist_arg**2,
        "bend": 0.5 * p.k33 * sin_sq * (d_theta**2 + cos_sq * d_phi**2),
        "e_z": e_z,
        # Orientation-dependent part of the dielectric energy density.
        "field": -0.5 * EPS0 * p.delta_eps * e_z**2 * np.sin(theta) ** 2,
    }


def free_energy_and_gradient(theta, phi, p: CellParams, voltage: float):
    """Discrete fixed-voltage potential per unit area and its node gradients.

    Returns (F in J/m^2, dF/dtheta_i, dF/dphi_i). Boundary entries of the
    gradients are computed but should be ignored (anchoring is fixed).
    """
    h = p.h
    theta_mid, d_theta, d_phi, cos_sq, sin_sq, twist_arg = _link_terms(theta, phi, p)
    sin_2t = np.sin(2 * theta_mid)

    f_links = (
        0.5 * p.k11 * cos_sq * d_theta**2
        + 0.5 * p.k22 * twist_arg**2
        + 0.5 * p.k33 * sin_sq * (d_theta**2 + cos_sq * d_phi**2)
    )
    df_dtheta_mid = sin_2t * (
        -0.5 * p.k11 * d_theta**2
        - p.k22 * twist_arg * d_phi
        + 0.5 * p.k33 * (d_theta**2 + np.cos(2 * theta_mid) * d_phi**2)
    )
    df_dd_theta = (p.k11 * cos_sq + p.k33 * sin_sq) * d_theta
    df_dd_phi = cos_sq * (p.k22 * twist_arg + p.k33 * sin_sq * d_phi)

    # Each link contributes h * f(theta_mid, d_theta, d_phi); chain rule onto the
    # two nodes it joins.
    grad_theta = np.zeros_like(theta)
    grad_theta[:-1] += 0.5 * h * df_dtheta_mid - df_dd_theta
    grad_theta[1:] += 0.5 * h * df_dtheta_mid + df_dd_theta
    grad_phi = np.zeros_like(phi)
    grad_phi[:-1] -= df_dd_phi
    grad_phi[1:] += df_dd_phi

    e_z, inverse_capacitance_integral = field_profile(theta, p, voltage)
    energy = h * f_links.sum() - 0.5 * EPS0 * voltage**2 / inverse_capacitance_integral
    grad_theta += (
        -0.5
        * h
        * _trapezoid_weights(theta.size)
        * EPS0
        * p.delta_eps
        * np.sin(2 * theta)
        * e_z**2
    )
    return energy, grad_theta, grad_phi


def stable_time_step(theta, p: CellParams, voltage: float) -> float:
    """Explicit-Euler time step for the gradient-flow dynamics, with safety margin."""
    elastic_rate = 4 * max(p.k11, p.k22, p.k33) / (p.h**2 * p.gamma1)
    e_z, _ = field_profile(theta, p, voltage)
    # Factor 2 on the field rate allows E to grow during a frame of switching.
    field_rate = 2 * EPS0 * abs(p.delta_eps) * np.max(e_z) ** 2 / p.gamma1
    return 1.0 / (elastic_rate + field_rate)


def relax_dynamics(
    theta, phi, p: CellParams, voltage: float, duration: float, max_steps: int
):
    """Integrate gamma1 dn/dt = -dF/dn (no backflow) for `duration` seconds.

    Uses torque balance in angle form:
        gamma1 dtheta/dt = -delta F / delta theta
        gamma1 cos^2(theta) dphi/dt = -delta F / delta phi
    Stops early after `max_steps`. Returns (theta, phi, time advanced, steps).
    """
    theta, phi = apply_boundaries(theta, phi, p)
    dt_max = stable_time_step(theta, p, voltage)
    n_steps = int(np.ceil(duration / dt_max))
    dt = duration / n_steps if n_steps > 0 else 0.0
    n_steps = min(n_steps, max_steps)
    rate = dt / (p.gamma1 * p.h)
    for _ in range(n_steps):
        _, grad_theta, grad_phi = free_energy_and_gradient(theta, phi, p, voltage)
        phi_mobility = 1.0 / (np.cos(theta[1:-1]) ** 2 + _PHI_MOBILITY_REGULARISATION)
        theta[1:-1] -= rate * grad_theta[1:-1]
        phi[1:-1] -= rate * phi_mobility * grad_phi[1:-1]
    return theta, phi, n_steps * dt, n_steps


def equilibrium(theta, phi, p: CellParams, voltage: float):
    """Minimise the fixed-voltage potential with L-BFGS, warm-started from a state.

    Finds the local minimum reached from the given state, so hysteresis and
    metastable branches are retained.
    """
    theta, phi = apply_boundaries(theta, phi, p)
    n_interior = p.n_nodes - 2
    # Make the objective O(1) so that L-BFGS tolerances are meaningful.
    scale = p.thickness / p.k11

    # The phi curvature scales as cos^2(theta), which is badly conditioned for
    # near-vertical directors. Optimise u = phi * cos(theta) instead. The scaling
    # is frozen within a pass, so passes are capped and the scaling is refreshed
    # until a pass converges (large tilt changes otherwise leave it stale).
    for _ in range(50):
        phi_scale = np.sqrt(np.cos(theta[1:-1]) ** 2 + _PHI_MOBILITY_REGULARISATION)

        def objective(x):
            theta_trial = theta.copy()
            phi_trial = phi.copy()
            theta_trial[1:-1] = x[:n_interior]
            phi_trial[1:-1] = x[n_interior:] / phi_scale
            energy, grad_theta, grad_phi = free_energy_and_gradient(
                theta_trial, phi_trial, p, voltage
            )
            return (
                energy * scale,
                np.concatenate([grad_theta[1:-1], grad_phi[1:-1] / phi_scale]) * scale,
            )

        result = minimize(
            objective,
            np.concatenate([theta[1:-1], phi[1:-1] * phi_scale]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 150, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-12},
        )
        theta[1:-1] = result.x[:n_interior]
        phi[1:-1] = result.x[n_interior:] / phi_scale
        if result.nit < 150:
            break
    return theta, phi


def freedericksz_threshold(p: CellParams):
    """Analytic threshold voltage (V) for zero pretilt and strong anchoring.

    Planar/twisted cells with delta_eps > 0:
        V_th = pi sqrt((K11 + (K33 - 2 K22) (Phi/pi)^2 + 2 K22 q0 d Phi / pi^2)
                       / (eps0 delta_eps))
    Homeotropic cells with delta_eps < 0:
        V_th = pi sqrt(K33 / (eps0 |delta_eps|))
    Returns None for other combinations (e.g. hybrid cells have no threshold).
    """
    is_planar = max(abs(p.pretilt_bottom), abs(p.pretilt_top)) < np.pi / 4
    is_homeotropic = min(abs(p.pretilt_bottom), abs(p.pretilt_top)) > np.pi / 4
    if is_planar and p.delta_eps > 0:
        twist_ratio = p.twist / np.pi
        k_eff = (
            p.k11
            + (p.k33 - 2 * p.k22) * twist_ratio**2
            + 2 * p.k22 * p.q0 * p.thickness * twist_ratio / np.pi
        )
    elif is_homeotropic and p.delta_eps < 0:
        k_eff = p.k33
    else:
        return None
    if k_eff <= 0:
        return None
    return np.pi * np.sqrt(k_eff / (EPS0 * abs(p.delta_eps)))


def layer_jones_matrices(theta, phi, p: CellParams, optics: OpticsParams):
    """Jones matrices (n_links, 2, 2) of each slab, index 0 at the bottom.

    Each slab is a linear retarder whose slow axis is the projected director
    azimuth and whose extraordinary index depends on the tilt for light along z.
    """
    theta_mid = 0.5 * (theta[:-1] + theta[1:])
    phi_mid = 0.5 * (phi[:-1] + phi[1:])
    n_e = optics.n_o + optics.delta_n
    n_eff = 1.0 / np.sqrt(
        np.sin(theta_mid) ** 2 / optics.n_o**2 + np.cos(theta_mid) ** 2 / n_e**2
    )
    retardance = 2 * np.pi * (n_eff - optics.n_o) * p.h / optics.wavelength
    slow = np.exp(-0.5j * retardance)
    fast = np.exp(0.5j * retardance)
    c = np.cos(phi_mid)
    s = np.sin(phi_mid)
    off_diagonal = (slow - fast) * c * s
    return np.stack(
        [
            np.stack([slow * c**2 + fast * s**2, off_diagonal], axis=-1),
            np.stack([off_diagonal, slow * s**2 + fast * c**2], axis=-1),
        ],
        axis=-2,
    )


def optical_outputs(theta, phi, p: CellParams, optics: OpticsParams):
    """Compute optical intensity output(s) for the cell and optics.

    Transmissive mode: light enters at z = 0 and exits at z = d.
    Reflective mode (LCoS-like): light enters at z = d, reflects off an ideal mirror at
    z = 0 and returns. By reciprocity, the return pass in the same lab frame is the
    transpose of the forward pass.

    Returns:
        If optics.analyser_angle is not None:
            a single float (fraction of intensity passing that analyser).
        If optics.analyser_angle is None:
            a tuple (t_parallel, t_crossed) of fractions for linear polarisations
            parallel and perpendicular to the input polariser azimuth.
    """
    matrices = layer_jones_matrices(theta, phi, p, optics)
    if optics.mode == "transmissive":
        ordered = matrices
    else:
        ordered = matrices[::-1]
    jones = np.eye(2, dtype=complex)
    for matrix in ordered:
        jones = matrix @ jones
    if optics.mode == "reflective":
        jones = jones.T @ jones

    alpha = optics.polarizer_angle
    e_in = np.array([np.cos(alpha), np.sin(alpha)])
    e_out = jones @ e_in

    if optics.analyser_angle is not None:
        beta = optics.analyser_angle
        e_analyser = np.array([np.cos(beta), np.sin(beta)])
        return float(abs(e_analyser @ e_out) ** 2)

    e_para = e_in
    e_cross = np.array([-np.sin(alpha), np.cos(alpha)])
    t_para = float(abs(e_para @ e_out) ** 2)
    t_cross = float(abs(e_cross @ e_out) ** 2)
    return t_para, t_cross


def optical_output(theta, phi, p: CellParams, optics: OpticsParams) -> float:
    """Fraction of polarised input intensity passing the analyser.

    If optics.analyser_angle is None, defaults to crossed polarisation.
    """
    out = optical_outputs(theta, phi, p, optics)
    if isinstance(out, tuple):
        return out[1]
    return out


def director_sweep(p: CellParams, voltages):
    """Equilibrium director profiles across a voltage sweep, warm-started."""
    theta, phi = initial_state(p)
    states = []
    for voltage in voltages:
        theta, phi = equilibrium(theta, phi, p, voltage)
        states.append((theta, phi))
    return states


def voltage_sweep(p: CellParams, optics: OpticsParams, voltages):
    """Quasi-static optical response, warm-starting each voltage from the last."""
    states = director_sweep(p, voltages)
    return np.array([optical_outputs(theta, phi, p, optics) for theta, phi in states])
