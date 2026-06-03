"""
solve (toy) problem on SO3 by lie algebra methods
"""

import matplotlib.pyplot as plt
import numpy as np


def solve_rotation_alignment(
    n: np.ndarray,
    target: np.ndarray,
    max_iters=100,
    tolerance=1e-9,
) -> np.ndarray:
    rot = np.eye(3)
    for _ in range(max_iters):
        # Compute residuals: e_i - R * n_i (shape: 3x3, each column is a residual vector)
        residuals = (target - rot @ n).flatten(order="F")

        # Compute the analytical Jacobian (9x3 matrix)
        # d/domega (R * exp(omega^) * n_i) evaluated at omega=0 is -R * (n_i)^
        # Therefore, d/domega (residual_i) = R * (n_i)^
        jacob = np.zeros((9, 3))
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
            jacob[i * 3 : (i + 1) * 3, :] = rot @ n_hat

        # Solve the Gauss-Newton normal equations: (J^T * J) * omega = -J^T * r
        omega = np.linalg.solve(jacob.T @ jacob, -jacob.T @ residuals)
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
        rot = rot @ exp_omega
    return rot


def plot_trivectors(*trivectors: np.ndarray) -> None:
    if not trivectors:
        raise ValueError("No trivectors provided!")
    fig = plt.figure(figsize=(7 * len(trivectors), 7))
    colors_target = ["#ff7675", "#55efc4", "#74b9ff"]
    colors_vec = ["#d63031", "#00b894", "#0984e3"]
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    for idx, trivector in enumerate(trivectors):
        ax = fig.add_subplot(1, len(trivectors), idx + 1, projection="3d")
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.4, linewidth=1)
        for i in range(3):
            # standard coordinates
            ax.quiver(
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
            )
            # trivectors
            ax.quiver(
                0,
                0,
                0,
                trivector[0, i],
                trivector[1, i],
                trivector[2, i],
                color=colors_vec[i],
                linewidth=2.5,
                arrow_length_ratio=0.1,
                label=f"Vector {i}",
            )

        ax.set_title(f"Trivector Set {idx + 1}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
    plt.tight_layout()


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 3 random normalized vectors (not orthogonal)
    raw_vectors = np.random.randn(3, 3)
    n_inputs = raw_vectors / np.linalg.norm(raw_vectors, axis=0)

    # Run optimization
    R_opt = solve_rotation_alignment(n_inputs, target=np.eye(3))

    print("Generated Vectors (n0, n1, n2):\n", n_inputs)
    print("\nOptimized Rotation Matrix R:\n", R_opt)
    print("\nCheck Orthogonality (R^T * R):\n", R_opt.T @ R_opt)

    result = R_opt @ n_inputs
    print("\nRotated Vectors (Should be close to Identity):\n", result)

    plot_trivectors(n_inputs, result)
    plt.show()
