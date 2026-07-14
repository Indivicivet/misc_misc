import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class FilterResult:
    name: str
    x_est: np.ndarray
    theoretical_pos_var: list
    errors: np.ndarray
    empirical_var: np.ndarray


@dataclass
class SimulationResult:
    t: np.ndarray
    x_true: np.ndarray
    z_meas: np.ndarray
    filter_results: list[FilterResult]
    dt: float
    n_steps: int
    total_steps: int


class KalmanFilter:
    def __init__(self, name="Kalman Filter"):
        self.name = name
        self.x = np.zeros(2)
        self.p_cov = np.eye(2)

    def reset(self):
        self.x = np.zeros(2)
        self.p_cov = np.eye(2)

    def compute_theoretical_limit_cycle(self, f_mat, q_mat, h_mat, r_mat, n_steps):
        p_temp = np.eye(2)
        for _ in range(100):
            for _ in range(n_steps):
                p_temp = f_mat @ p_temp @ f_mat.T + q_mat
            s_mat = h_mat @ p_temp @ h_mat.T + r_mat
            k_gain = p_temp @ h_mat.T @ np.linalg.inv(s_mat)
            p_temp = (np.eye(2) - k_gain @ h_mat) @ p_temp

        cycle = []
        for _ in range(n_steps - 1):
            p_temp = f_mat @ p_temp @ f_mat.T + q_mat
            cycle.append(p_temp.copy())

        p_temp = f_mat @ p_temp @ f_mat.T + q_mat
        cycle.append(p_temp.copy())

        s_mat = h_mat @ p_temp @ h_mat.T + r_mat
        k_opt = p_temp @ h_mat.T @ np.linalg.inv(s_mat)
        p_post_update = (np.eye(2) - k_opt @ h_mat) @ p_temp

        cycle_plot = cycle[:-1] + [p_post_update]
        return [p[0, 0] for p in cycle_plot]

    def predict(self, u_in, f_mat, b_mat, q_mat):
        self.x = f_mat @ self.x + (b_mat * u_in).flatten()
        self.p_cov = f_mat @ self.p_cov @ f_mat.T + q_mat

    def update(self, z_in, h_mat, r_mat):
        s_mat = h_mat @ self.p_cov @ h_mat.T + r_mat
        k_gain = self.p_cov @ h_mat.T @ np.linalg.inv(s_mat)
        y_err = z_in - (h_mat @ self.x)[0]
        self.x = self.x + (k_gain * y_err).flatten()
        self.p_cov = (np.eye(2) - k_gain @ h_mat) @ self.p_cov


class ComplementaryFilter:
    def __init__(self, name="Complementary Filter", k_gain=None):
        self.name = name
        self.x = np.zeros(2)
        self.k_gain = k_gain

    def reset(self):
        self.x = np.zeros(2)

    def compute_theoretical_limit_cycle(self, f_mat, q_mat, h_mat, r_mat, n_steps):
        # Calculate optimal gain to scale from
        p_temp = np.eye(2)
        for _ in range(100):
            for _ in range(n_steps):
                p_temp = f_mat @ p_temp @ f_mat.T + q_mat
            s_mat = h_mat @ p_temp @ h_mat.T + r_mat
            k_opt = p_temp @ h_mat.T @ np.linalg.inv(s_mat)
            p_temp = (np.eye(2) - k_opt @ h_mat) @ p_temp

        if self.k_gain is None:
            self.k_gain = k_opt * 0.3

        p_cf = np.eye(2)
        for _ in range(100):
            for _ in range(n_steps):
                p_cf = f_mat @ p_cf @ f_mat.T + q_mat
            i_kh = np.eye(2) - self.k_gain @ h_mat
            p_cf = i_kh @ p_cf @ i_kh.T + self.k_gain @ r_mat @ self.k_gain.T

        cycle = []
        for _ in range(n_steps - 1):
            p_cf = f_mat @ p_cf @ f_mat.T + q_mat
            cycle.append(p_cf.copy())

        p_cf = f_mat @ p_cf @ f_mat.T + q_mat
        cycle.append(p_cf.copy())

        i_kh = np.eye(2) - self.k_gain @ h_mat
        p_post_update = i_kh @ p_cf @ i_kh.T + self.k_gain @ r_mat @ self.k_gain.T

        cycle_plot = cycle[:-1] + [p_post_update]
        return [p[0, 0] for p in cycle_plot]

    def predict(self, u_in, f_mat, b_mat, q_mat):
        self.x = f_mat @ self.x + (b_mat * u_in).flatten()

    def update(self, z_in, h_mat, r_mat):
        y_err = z_in - (h_mat @ self.x)[0]
        self.x = self.x + (self.k_gain * y_err).flatten()


def run_simulation(
    algorithms=None, dt=0.01, n_steps=10, sigma_a=0.5, sigma_p=2.0, num_cycles=1000
) -> SimulationResult:
    if algorithms is None:
        algorithms = [KalmanFilter(), ComplementaryFilter()]

    # System matrices
    f_mat = np.array([[1, dt], [0, 1]])
    b_mat = np.array([[0.5 * dt**2], [dt]])
    h_mat = np.array([[1, 0]])

    # Covariance matrices
    q_mat = (sigma_a**2) * (b_mat @ b_mat.T)
    r_mat = np.array([[sigma_p**2]])

    np.random.seed(42)
    total_steps = num_cycles * n_steps

    t_arr = np.arange(total_steps) * dt
    a_true = 2.0 * np.sin(0.5 * t_arr)

    x_true = np.zeros(total_steps)
    v_true = np.zeros(total_steps)
    for i in range(1, total_steps):
        v_true[i] = v_true[i - 1] + a_true[i - 1] * dt
        x_true[i] = x_true[i - 1] + v_true[i - 1] * dt + 0.5 * a_true[i - 1] * dt**2

    a_meas = a_true + np.random.normal(0, sigma_a, total_steps)
    z_meas = x_true + np.random.normal(0, sigma_p, total_steps)

    filter_results = []

    for algo in algorithms:
        algo.reset()
        theoretical_pos_var = algo.compute_theoretical_limit_cycle(
            f_mat, q_mat, h_mat, r_mat, n_steps
        )

        x_est = np.zeros((2, total_steps))
        errors = np.zeros((num_cycles, n_steps))

        for c in range(num_cycles):
            for i in range(n_steps):
                step_idx = c * n_steps + i
                if step_idx == 0:
                    continue

                u_in = a_meas[step_idx - 1]
                algo.predict(u_in, f_mat, b_mat, q_mat)

                if i == n_steps - 1:
                    algo.update(z_meas[step_idx], h_mat, r_mat)

                x_est[:, step_idx] = algo.x
                errors[c, i] = algo.x[0] - x_true[step_idx]

        empirical_var = np.var(errors[50:, :], axis=0)

        filter_results.append(
            FilterResult(
                name=algo.name,
                x_est=x_est,
                theoretical_pos_var=theoretical_pos_var,
                errors=errors,
                empirical_var=empirical_var,
            )
        )

    return SimulationResult(
        t=t_arr,
        x_true=x_true,
        z_meas=z_meas,
        filter_results=filter_results,
        dt=dt,
        n_steps=n_steps,
        total_steps=total_steps,
    )


if __name__ == "__main__":
    res = run_simulation()

    # Plotting
    plt.figure(figsize=(12, 10))

    colors_traj = ["b-", "g-", "m-", "c-"]
    colors_var = ["b", "r", "m", "c"]
    marker_types = ["o", "x", "s", "^"]

    # Plot 1: True trajectory and noisy measurements
    plt.subplot(2, 2, 1)
    plt.plot(res.t, res.x_true, "k-", label="True Position", linewidth=2)
    gps_idx = np.arange(res.total_steps)
    gps_idx = gps_idx[gps_idx % res.n_steps == res.n_steps - 1]
    plt.plot(
        res.t[gps_idx],
        res.z_meas[gps_idx],
        "ro",
        label="Noisy GPS",
        markersize=2,
        alpha=0.5,
    )

    for i, f_res in enumerate(res.filter_results):
        c_traj = colors_traj[i % len(colors_traj)]
        plt.plot(res.t, f_res.x_est[0], c_traj, label=f_res.name)

    plt.title("Full Trajectory (Position)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    # Plot 2: Variance Limit Cycle
    plt.subplot(2, 2, 2)
    cycle_time = np.arange(1, res.n_steps + 1) * res.dt

    for i, f_res in enumerate(res.filter_results):
        c_var = colors_var[i % len(colors_var)]
        m_type = marker_types[i % len(marker_types)]
        plt.plot(
            cycle_time,
            f_res.theoretical_pos_var,
            f"{c_var}-",
            label=f"{f_res.name} Theoretical",
            linewidth=2,
        )
        plt.plot(
            cycle_time,
            f_res.empirical_var,
            f"{c_var}--",
            label=f"{f_res.name} Empirical",
            marker=m_type,
            markersize=5,
        )

    plt.title(f"Position Variance Limit Cycle (N={res.n_steps})")
    plt.xlabel("Time since last position update (s)")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)

    # Plot 3: Error Distributions (Post-update)
    plt.subplot(2, 2, 3)
    for i, f_res in enumerate(res.filter_results):
        c_var = colors_var[i % len(colors_var)]
        plt.hist(
            f_res.errors[50:, -1],
            bins=30,
            alpha=0.5,
            color=c_var,
            label=f_res.name,
            density=True,
        )
    plt.title("Position Error Distribution\n(Post-update)")
    plt.xlabel("Position Error")
    plt.ylabel("Density")
    plt.legend()

    # Plot 4: Error Distributions (Pre-update)
    plt.subplot(2, 2, 4)
    for i, f_res in enumerate(res.filter_results):
        c_var = colors_var[i % len(colors_var)]
        plt.hist(
            f_res.errors[50:, res.n_steps - 2],
            bins=30,
            alpha=0.5,
            color=c_var,
            label=f_res.name,
            density=True,
        )
    plt.title("Position Error Distribution\n(Pre-update)")
    plt.xlabel("Position Error")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.show()
