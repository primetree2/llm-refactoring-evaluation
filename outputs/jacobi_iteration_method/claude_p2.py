"""Jacobi iteration method for solving systems of linear equations."""

import numpy as np


def jacobi_iteration_method(
    coefficient_matrix: np.ndarray,
    constant_matrix: np.ndarray,
    initial_values: list[float],
    num_iterations: int,
) -> list[float]:
    """
    Solve a system of linear equations using the Jacobi iteration method.

    Given a system Ax = b, this method iteratively approximates the solution
    vector x, starting from an initial guess.

    Args:
        coefficient_matrix: Square matrix (n x n) of coefficients (A).
        constant_matrix: Column vector (n x 1) of constants (b).
        initial_values: List of initial guess values for the solution.
        num_iterations: Number of iterations to perform (must be >= 1).

    Returns:
        List of approximate solution values after the specified iterations.

    Raises:
        ValueError: If matrix dimensions are incompatible, initial values
            count doesn't match, iterations is non-positive, or the
            coefficient matrix is not strictly diagonally dominant.
    """
    _validate_inputs(coefficient_matrix, constant_matrix, initial_values, num_iterations)

    # Build augmented matrix [A | b]
    augmented = np.concatenate((coefficient_matrix, constant_matrix), axis=1)
    num_rows, num_cols = augmented.shape

    # Ensure convergence condition is met
    _check_strictly_diagonally_dominant(augmented)

    current_values = list(initial_values)

    for _ in range(num_iterations):
        next_values = []

        for i in range(num_rows):
            # Sum contributions from non-diagonal coefficient terms
            off_diagonal_sum = sum(
                -augmented[i][j] * current_values[j]
                for j in range(num_cols - 1)
                if j != i
            )

            # Compute new value: x_i = (b_i - sum(a_ij * x_j, j != i)) / a_ii
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
    Validate the inputs for the Jacobi iteration method.

    Raises:
        ValueError: If any input dimensions or values are invalid.
    """
    coeff_rows, coeff_cols = coefficient_matrix.shape
    const_rows, const_cols = constant_matrix.shape

    if coeff_rows != coeff_cols:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn "
            f"but received {coeff_rows}x{coeff_cols}"
        )

    if const_cols != 1:
        raise ValueError(
            f"Constant matrix must be nx1 "
            f"but received {const_rows}x{const_cols}"
        )

    if coeff_rows != const_rows:
        raise ValueError(
            f"Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {coeff_rows}x{coeff_cols} and {const_rows}x{const_cols}"
        )

    if len(initial_values) != coeff_rows:
        raise ValueError(
            f"Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(initial_values)} and {coeff_rows}"
        )

    if num_iterations <= 0:
        raise ValueError("Iterations must be at least 1")


def _check_strictly_diagonally_dominant(augmented: np.ndarray) -> None:
    """
    Check that the coefficient portion of the augmented matrix is strictly
    diagonally dominant.

    Note:
        This preserves the original comparison logic, which does not use
        absolute values for off-diagonal elements.

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
