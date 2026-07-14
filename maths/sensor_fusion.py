import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.01  # Accelerometer sampling time
n_steps = 10  # Ratio of accelerometer rate to position sensor rate
dt_large = n_steps * dt

sigma_a = 0.5  # Accelerometer noise std dev (process noise)
sigma_p = 2.0  # Position sensor noise std dev (measurement noise)

# System matrices
# State = [position, velocity]^T
f = np.array([[1, dt], [0, 1]])
b = np.array([[0.5 * dt**2], [dt]])
h = np.array([[1, 0]])

# Covariance matrices
q = (sigma_a**2) * (b @ b.T)
r = np.array([[sigma_p**2]])

# 1. Theoretical Covariance Computation (Limit Cycle)
# ----------------------------------------------------
# We find the steady-state limit cycle of the Kalman Filter covariance
p_kf = np.eye(2)  # Initial covariance

# Run enough cycles to reach steady state
for _ in range(100):
    for _ in range(n_steps):
        p_kf = f @ p_kf @ f.T + q
    # Update
    s = h @ p_kf @ h.T + r
    k = p_kf @ h.T @ np.linalg.inv(s)
    p_kf = (np.eye(2) - k @ h) @ p_kf

# Now record one full steady-state cycle
p_kf_cycle = []
for _ in range(n_steps - 1):
    p_kf = f @ p_kf @ f.T + q
    p_kf_cycle.append(p_kf.copy())

# Last prediction step before update
p_kf = f @ p_kf @ f.T + q
p_kf_cycle.append(p_kf.copy())

# Update at the end of the cycle (not strictly part of the "prediction" array
# but let's record the post-update covariance as the FIRST element of the NEXT cycle logically.
# To plot, we want to see the error grow during predictions and drop at the end)
s = h @ p_kf @ h.T + r
k_opt = p_kf @ h.T @ np.linalg.inv(s)
p_kf_post_update = (np.eye(2) - k_opt @ h) @ p_kf

# We will plot the cycle ending with the post-update collapse.
p_kf_cycle_plot = p_kf_cycle[:-1] + [p_kf_post_update]

# Compute theoretical cycle for a Complementary Filter (Suboptimal Gain)
# Let's use a gain that is heavily detuned to clearly see suboptimal performance
k_cf = k_opt * 0.3
p_cf = np.eye(2)
for _ in range(100):
    for _ in range(n_steps):
        p_cf = f @ p_cf @ f.T + q
    # Suboptimal update equation: p+ = (I-kh)p-(I-kh)' + k r k'
    i_kh = np.eye(2) - k_cf @ h
    p_cf = i_kh @ p_cf @ i_kh.T + k_cf @ r @ k_cf.T

p_cf_cycle = []
for _ in range(n_steps - 1):
    p_cf = f @ p_cf @ f.T + q
    p_cf_cycle.append(p_cf.copy())

p_cf = f @ p_cf @ f.T + q
p_cf_cycle.append(p_cf.copy())

i_kh = np.eye(2) - k_cf @ h
p_cf_post_update = i_kh @ p_cf @ i_kh.T + k_cf @ r @ k_cf.T
p_cf_cycle_plot = p_cf_cycle[:-1] + [p_cf_post_update]

theoretical_kf_pos_var = [p[0, 0] for p in p_kf_cycle_plot]
theoretical_cf_pos_var = [p[0, 0] for p in p_cf_cycle_plot]

# 2. Simulation
# ----------------------------------------------------
np.random.seed(42)
num_cycles = 1000
total_steps = num_cycles * n_steps

# True trajectory (sine wave acceleration)
t = np.arange(total_steps) * dt
a_true = 2.0 * np.sin(0.5 * t)

x_true = np.zeros(total_steps)
v_true = np.zeros(total_steps)
for i in range(1, total_steps):
    v_true[i] = v_true[i - 1] + a_true[i - 1] * dt
    x_true[i] = x_true[i - 1] + v_true[i - 1] * dt + 0.5 * a_true[i - 1] * dt**2

# Generate noisy measurements
a_meas = a_true + np.random.normal(0, sigma_a, total_steps)
z_meas = x_true + np.random.normal(0, sigma_p, total_steps)

# Run Filters
x_kf = np.zeros((2, total_steps))
x_cf = np.zeros((2, total_steps))

p_kf_sim = np.eye(2)

# Variables to store cycle errors for empirical variance
kf_errors = np.zeros((num_cycles, n_steps))
cf_errors = np.zeros((num_cycles, n_steps))

for c in range(num_cycles):
    for i in range(n_steps):
        step_idx = c * n_steps + i
        if step_idx == 0:
            continue

        # Prediction
        u = a_meas[step_idx - 1]
        x_kf[:, step_idx] = f @ x_kf[:, step_idx - 1] + (b * u).flatten()
        p_kf_sim = f @ p_kf_sim @ f.T + q

        x_cf[:, step_idx] = f @ x_cf[:, step_idx - 1] + (b * u).flatten()

        # Update (only on the last step of the cycle)
        if i == n_steps - 1:
            # KF Update
            s_sim = h @ p_kf_sim @ h.T + r
            k_sim = p_kf_sim @ h.T @ np.linalg.inv(s_sim)
            y = z_meas[step_idx] - (h @ x_kf[:, step_idx])[0]
            x_kf[:, step_idx] = x_kf[:, step_idx] + (k_sim * y).flatten()
            p_kf_sim = (np.eye(2) - k_sim @ h) @ p_kf_sim

            # CF Update
            y_cf = z_meas[step_idx] - (h @ x_cf[:, step_idx])[0]
            x_cf[:, step_idx] = x_cf[:, step_idx] + (k_cf * y_cf).flatten()

        # Record errors
        kf_errors[c, i] = x_kf[0, step_idx] - x_true[step_idx]
        cf_errors[c, i] = x_cf[0, step_idx] - x_true[step_idx]

# Compute empirical variance across cycles
# Ignore first 50 cycles for burn-in
empirical_kf_var = np.var(kf_errors[50:, :], axis=0)
empirical_cf_var = np.var(cf_errors[50:, :], axis=0)

# 3. Plotting
# ----------------------------------------------------
plt.figure(figsize=(12, 10))

# Plot 1: True trajectory and noisy measurements (full)
plt.subplot(2, 2, 1)
plt.plot(t, x_true, "k-", label="True Position", linewidth=2)
# Plot GPS points (every n_steps'th step)
gps_idx = np.arange(total_steps)
gps_idx = gps_idx[gps_idx % n_steps == n_steps - 1]
plt.plot(t[gps_idx], z_meas[gps_idx], "ro", label="Noisy GPS", markersize=2, alpha=0.5)
plt.plot(t, x_kf[0], "b-", label="Kalman Filter")
plt.plot(t, x_cf[0], "g-", label="Complementary Filter")
plt.title("Full Trajectory (Position)")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

# Plot 2: Theoretical vs Empirical Variance (Limit Cycle)
plt.subplot(2, 2, 2)
cycle_time = np.arange(1, n_steps + 1) * dt
plt.plot(
    cycle_time,
    theoretical_kf_pos_var,
    "b-",
    label="Kalman Filter Theoretical",
    linewidth=2,
)
plt.plot(
    cycle_time,
    empirical_kf_var,
    "b--",
    label="Kalman Filter Empirical",
    marker="o",
    markersize=5,
)

plt.plot(
    cycle_time,
    theoretical_cf_pos_var,
    "r-",
    label="Complementary Filter Theoretical",
    linewidth=2,
)
plt.plot(
    cycle_time,
    empirical_cf_var,
    "r--",
    label="Complementary Filter Empirical",
    marker="x",
    markersize=5,
)

plt.title(f"Position Variance Limit Cycle (N={n_steps})")
plt.xlabel("Time since last position update (s)")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)

# Plot 3: Tracking Error distributions
plt.subplot(2, 2, 3)
plt.hist(
    kf_errors[50:, -1],
    bins=30,
    alpha=0.5,
    color="b",
    label="Kalman Filter",
    density=True,
)
plt.hist(
    cf_errors[50:, -1],
    bins=30,
    alpha=0.5,
    color="r",
    label="Complementary Filter",
    density=True,
)
plt.title("Position Error Distribution\n(Post-update)")
plt.xlabel("Position Error")
plt.ylabel("Density")
plt.legend()

plt.subplot(2, 2, 4)
# Pre-update error is at index n_steps-2
plt.hist(
    kf_errors[50:, n_steps - 2],
    bins=30,
    alpha=0.5,
    color="b",
    label="Kalman Filter",
    density=True,
)
plt.hist(
    cf_errors[50:, n_steps - 2],
    bins=30,
    alpha=0.5,
    color="r",
    label="Complementary Filter",
    density=True,
)
plt.title("Position Error Distribution\n(Pre-update)")
plt.xlabel("Position Error")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()

# Check if empirical and theoretical match
kf_error_diff = np.max(np.abs(empirical_kf_var - theoretical_kf_pos_var))
print(
    f"Max difference between empirical and theoretical KF variance:"
    f" {kf_error_diff:.4f}"
)
