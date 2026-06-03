"""
solve (toy) problem on SO3 by lie algebra methods
"""

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

        # 3. Solve the Gauss-Newton normal equations: (J^T * J) * omega = J^T * r
        omega = np.linalg.solve(J.T @ J, J.T @ r)

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


# --- Verification ---
np.random.seed(42)

# Generate 3 random normalized vectors (not orthogonal)
raw_vectors = np.random.randn(3, 3)
n_inputs = raw_vectors / np.linalg.norm(raw_vectors, axis=0)

# Run optimization
R_opt = solve_rotation_alignment(n_inputs)

print("Generated Vectors (n0, n1, n2):\n", n_inputs)
print("\nOptimized Rotation Matrix R:\n", R_opt)
print("\nCheck Orthogonality (R^T * R):\n", R_opt.T @ R_opt)
print("\nRotated Vectors (Should be close to Identity):\n", R_opt @ n_inputs)