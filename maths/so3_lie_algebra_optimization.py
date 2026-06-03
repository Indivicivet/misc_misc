"""
solve (toy) problem on SO3 by lie algebra methods
"""

import matplotlib.pyplot as plt
import numpy as np


def solve_rotation_alignment(n: np.ndarray) -> np.ndarray:
    # Target standard basis vectors (e_0, e_1, e_2)
    E = np.eye(3)

    # Initialize R as identity matrix
    R = np.eye(3)

    max_iters = 100
    tolerance = 1e-9

    for _ in range(max_iters):
        # 1. Compute residuals: e_i - R * n_i (shape: 3x3, each column is a residual vector)
        residuals = E - R @ n
        r = residuals.flatten(order="F")  # Flatten to a 9-dimensional column vector

        # 2. Compute the analytical Jacobian (9x3 matrix)
        # d/domega (R * exp(omega^) * n_i) evaluated at omega=0 is -R * (n_i)^
        # Therefore, d/domega (residual_i) = R * (n_i)^
        J = np.zeros((9, 3))
        for i in range(3):
            n_i = n[:, i]
            # Skew-symmetric matrix (hat operator) for n_i
            n_hat = np.array(
                [
                    [0.0, -n_i[2], n_i[1]],
                    [n_i[2], 0.0, -n_i[0]],
                    [-n_i[1], n_i[0], 0.0],
                ]
            )
            J[i * 3 : (i + 1) * 3, :] = R @ n_hat

        # 3. Solve the Gauss-Newton normal equations: (J^T * J) * omega = -J^T * r
        omega = np.linalg.solve(J.T @ J, -J.T @ r)

        # 4. Retraction: Update R using Rodrigues' formula for exp(omega^)
        theta = np.linalg.norm(omega)
        if theta < tolerance:
            break

        omega_hat = np.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ]
        )

        # Rodrigues' rotation formula
        exp_omega = (
            np.eye(3)
            + (np.sin(theta) / theta) * omega_hat
            + ((1.0 - np.cos(theta)) / (theta**2)) * (omega_hat @ omega_hat)
        )

        R = R @ exp_omega

    return R


def plot_results(vectors0, vectors1):
    fig = plt.figure(figsize=(14, 7))

    # Define colors and labels matching standard coordinate axes (X: Red, Y: Green, Z: Blue)
    colors_target = ["#ff7675", "#55efc4", "#74b9ff"]
    labels_target = [r"Target $\hat{x}$", r"Target $\hat{y}$", r"Target $\hat{z}$"]
    colors_vec = ["#d63031", "#00b894", "#0984e3"]
    labels_orig = [r"Original $n_0$", r"Original $n_1$", r"Original $n_2$"]
    labels_rot = [r"Rotated $n_0$", r"Rotated $n_1$", r"Rotated $n_2$"]

    # Faint unit sphere for 3D perspective context
    u, v = np.mgrid[0: 2 * np.pi: 20j, 0: np.pi: 10j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    # 1. Left Plot: Original Vectors vs Targets
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_wireframe(xs, ys, zs, color="gray", alpha=0.1, linewidth=0.5)

    # Plot Targets (dashed)
    for i in range(3):
        ax1.quiver(
            0,
            0,
            0,
            1 if i == 0 else 0,
            1 if i == 1 else 0,
            1 if i == 2 else 0,
            color=colors_target[i],
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            arrow_length_ratio=0.1,
            label=labels_target[i],
        )

    # Plot Original Vectors
    for i in range(3):
        ax1.quiver(
            0,
            0,
            0,
            vectors0[0, i],
            vectors0[1, i],
            vectors0[2, i],
            color=colors_vec[i],
            linewidth=2.5,
            arrow_length_ratio=0.1,
            label=labels_orig[i],
        )

    ax1.set_title("Original Vectors vs. Targets", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(loc="upper left")

    # 2. Right Plot: Rotated Vectors vs Targets
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_wireframe(xs, ys, zs, color="gray", alpha=0.1, linewidth=0.5)

    # Plot Targets (dashed)
    for i in range(3):
        ax2.quiver(
            0,
            0,
            0,
            1 if i == 0 else 0,
            1 if i == 1 else 0,
            1 if i == 2 else 0,
            color=colors_target[i],
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            arrow_length_ratio=0.1,
            label=labels_target[i],
        )

    # Plot Rotated Vectors
    for i in range(3):
        ax2.quiver(
            0,
            0,
            0,
            vectors1[0, i],
            vectors1[1, i],
            vectors1[2, i],
            color=colors_vec[i],
            linewidth=2.5,
            arrow_length_ratio=0.1,
            label=labels_rot[i],
        )

    ax2.set_title("Rotated Vectors vs. Targets", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 3 random normalized vectors (not orthogonal)
    raw_vectors = np.random.randn(3, 3)
    n_inputs = raw_vectors / np.linalg.norm(raw_vectors, axis=0)

    # Run optimization
    R_opt = solve_rotation_alignment(n_inputs)

    print("Generated Vectors (n0, n1, n2):\n", n_inputs)
    print("\nOptimized Rotation Matrix R:\n", R_opt)
    print("\nCheck Orthogonality (R^T * R):\n", R_opt.T @ R_opt)

    result = R_opt @ n_inputs
    print("\nRotated Vectors (Should be close to Identity):\n", result)

    plot_results(n_inputs, result)
