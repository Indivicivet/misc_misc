"""
solve (toy) problem on SO3 by lie algebra methods
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def solve_rotation_alignment(
    n: np.ndarray,
    target: np.ndarray,
    max_iters=100,
    tolerance=1e-9,
    return_history=False,
) -> tuple[np.ndarray, list[np.ndarray]] | np.ndarray:
    rot = np.eye(3)
    rot_history = [rot.copy()]
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
        rot_history.append(rot.copy())

    if return_history:
        return rot, rot_history
    return rot


def rotation_matrix_to_vector(R: np.ndarray) -> np.ndarray:
    """Map a 3x3 rotation matrix to its Lie algebra vector representation.

    Uses the SO(3) logarithmic map.
    """
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if np.abs(theta) < 1e-10:
        return np.zeros(3)
    if np.abs(theta - np.pi) < 1e-6:
        # Find the axis from the symmetric part of R
        v_sq = (R + np.eye(3)) / 2.0
        i = np.argmax(np.diag(v_sq))
        v = v_sq[:, i] / np.sqrt(v_sq[i, i])
        return v * theta
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def plot_trivectors(
    *trivectors: np.ndarray,
    history: Optional[list[np.ndarray]] = None,
    initial_vecs: Optional[np.ndarray] = None,
    plot_lie_algebra: bool = True,
) -> None:
    if not trivectors:
        raise ValueError("No trivectors provided!")
    if plot_lie_algebra and history is None:
        raise ValueError("Need to provide history to plot lie algebra")
    num_plots = len(trivectors)
    if plot_lie_algebra:
        num_plots += 1
    fig = plt.figure(figsize=(7 * num_plots, 7))
    colors_target = ["#ff7675", "#55efc4", "#74b9ff"]
    colors_vec = ["#d63031", "#00b894", "#0984e3"]
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    for idx, trivector in enumerate(trivectors):
        ax = fig.add_subplot(1, num_plots, idx + 1, projection="3d")
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

        # Plot paths if history is provided
        if idx > 0 and history is not None and initial_vecs is not None:
            for i in range(3):
                path = np.array([R @ initial_vecs[:, i] for R in history])
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    color=colors_vec[i],
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.scatter(
                    path[:, 0],
                    path[:, 1],
                    path[:, 2],
                    color=colors_vec[i],
                    s=20,
                    depthshade=False,
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

    if plot_lie_algebra:
        ax = fig.add_subplot(1, num_plots, num_plots, projection="3d")
        # Calculate Lie algebra representation for each rotation matrix in history
        # todo :: maybe nicer if this were included in the history instead
        lie_path = np.array([rotation_matrix_to_vector(R) for R in history])
        ax.plot(
            lie_path[:, 0],
            lie_path[:, 1],
            lie_path[:, 2],
            color="#6c5ce7",
            linestyle="-",
            linewidth=2.5,
            label="Optimization Path",
        )
        ax.scatter(
            lie_path[:, 0],
            lie_path[:, 1],
            lie_path[:, 2],
            color="#a29bfe",
            s=30,
            depthshade=False,
            edgecolors="#6c5ce7",
            linewidths=1.0,
            label="Iterations",
        )
        ax.scatter(
            [0],
            [0],
            [0],
            color="#ff7675",
            s=80,
            marker="*",
            depthshade=False,
            edgecolors="black",
            linewidths=1.0,
            label="Start (Identity)",
            zorder=10,
        )
        ax.scatter(
            [lie_path[-1, 0]],
            [lie_path[-1, 1]],
            [lie_path[-1, 2]],
            color="#55efc4",
            s=80,
            marker="D",
            depthshade=False,
            edgecolors="black",
            linewidths=1.0,
            label="Final Rotation",
            zorder=10,
        )
        # Draw coordinate axes through the origin
        max_val = np.max(np.abs(lie_path))
        axis_limit = max(max_val * 1.2, 0.5)
        ax.plot(
            [-axis_limit, axis_limit],
            [0, 0],
            [0, 0],
            color="black",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(
            [0, 0],
            [-axis_limit, axis_limit],
            [0, 0],
            color="black",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(
            [0, 0],
            [0, 0],
            [-axis_limit, axis_limit],
            color="black",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
        )
        ax.set_title(
            "Path in Lie Algebra $\\mathfrak{so}(3)$",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("$\\omega_x$ (rad)")
        ax.set_ylabel("$\\omega_y$ (rad)")
        ax.set_zlabel("$\\omega_z$ (rad)")
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
    plt.tight_layout()


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 3 random normalized vectors (not orthogonal)
    raw_vectors = np.random.randn(3, 3)
    random_unit_vecs = raw_vectors / np.linalg.norm(raw_vectors, axis=0)

    # Run optimization with history tracking
    R_opt, rot_history = solve_rotation_alignment(
        random_unit_vecs,
        target=np.eye(3),
        return_history=True,
    )

    print("Generated Vectors (n0, n1, n2):\n", random_unit_vecs)
    print("\nOptimized Rotation Matrix R:\n", R_opt)
    print("\nCheck Orthogonality (R^T * R):\n", R_opt.T @ R_opt)

    result = R_opt @ random_unit_vecs
    print("\nRotated Vectors (Should be close to Identity):\n", result)

    plot_trivectors(
        random_unit_vecs,
        result,
        history=rot_history,
        initial_vecs=random_unit_vecs,
    )
    plt.show()
