"""Module implementing the Jacobi iteration method for solving linear systems."""

import numpy as np


def jacobi_iteration_method(
    coefficient_matrix: np.ndarray,
    constant_matrix: np.ndarray,
    initial_values: list[float],
    num_iterations: int,
) -> list[float]:
    """
    Solve a system of linear equations using the Jacobi iteration method.

    The Jacobi method iteratively approximates the solution to a system Ax = b,
    where A is a strictly diagonally dominant coefficient matrix.

    Args:
        coefficient_matrix: An n×n numpy array representing the coefficient matrix (A).
        constant_matrix: An n×1 numpy array representing the constant vector (b).
        initial_values: A list of initial guesses for the solution variables.
        num_iterations: The number of iterations to perform (must be >= 1).

    Returns:
        A list of approximated solution values after the specified iterations.

    Raises:
        ValueError: If matrix dimensions are incompatible, iteration count is
            invalid, or the coefficient matrix is not strictly diagonally dominant.
    """
    _validate_inputs(coefficient_matrix, constant_matrix, initial_values, num_iterations)

    # Build augmented matrix [A | b] for convenience
    augmented = np.concatenate((coefficient_matrix, constant_matrix), axis=1)
    num_rows, num_cols = augmented.shape

    # Verify convergence condition
    _check_strictly_diagonally_dominant(augmented)

    current_values = list(initial_values)

    for _ in range(num_iterations):
        next_values = []

        for i in range(num_rows):
            # Sum contributions from off-diagonal elements
            off_diagonal_sum = sum(
                -augmented[i][j] * current_values[j]
                for j in range(num_cols - 1)
                if j != i
            )

            # Compute new value: x_i = (b_i - sum(a_ij * x_j)) / a_ii
            diagonal = augmented[i][i]
            constant = augmented[i][num_cols - 1]
            next_values.append((off_diagonal_sum + constant) / diagonal)

        current_values = next_values

    return current_values


def _validate_inputs(
    coefficient_matrix: np.ndarray,
    constant_matrix: np.ndarray,
    initial_values: list[float],
    num_iterations: int,
) -> None:
    """
    Validate inputs for the Jacobi iteration method.

    Raises:
        ValueError: If any input dimensions or values are invalid.
    """
    n_rows, n_cols = coefficient_matrix.shape
    b_rows, b_cols = constant_matrix.shape

    if n_rows != n_cols:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received {n_rows}x{n_cols}"
        )

    if b_cols != 1:
        raise ValueError(
            f"Constant matrix must be nx1 but received {b_rows}x{b_cols}"
        )

    if n_rows != b_rows:
        raise ValueError(
            f"Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {n_rows}x{n_cols} and {b_rows}x{b_cols}"
        )

    if len(initial_values) != n_rows:
        raise ValueError(
            f"Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(initial_values)} and {n_rows}"
        )

    if num_iterations <= 0:
        raise ValueError("Iterations must be at least 1")


def _check_strictly_diagonally_dominant(augmented: np.ndarray) -> None:
    """
    Check that the coefficient portion of the augmented matrix is strictly
    diagonally dominant.

    A matrix is strictly diagonally dominant if for every row, the diagonal
    element is greater than the sum of the other elements in that row.

    Note: This preserves the original behavior which does not use absolute
    values in the comparison.

    Args:
        augmented: The augmented matrix [A | b].

    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.
    """
    num_rows, num_cols = augmented.shape

    for i in range(num_rows):
        off_diagonal_sum = sum(
            augmented[i][j]
            for j in range(num_cols - 1)
            if j != i
        )

        if augmented[i][i] <= off_diagonal_sum:
            raise ValueError(
                "Coefficient matrix is not strictly diagonally dominant"
            )


if __name__ == "__main__":
    import doctest

    result = doctest.testmod()
    print(result)
