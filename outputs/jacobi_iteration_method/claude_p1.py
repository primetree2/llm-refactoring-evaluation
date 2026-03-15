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

    Args:
        coefficient_matrix: An n×n matrix of coefficients.
        constant_matrix: An n×1 matrix of constants.
        initial_values: Initial guess values for the solution.
        num_iterations: Number of iterations to perform.

    Returns:
        A list of approximate solution values after the specified iterations.

    Raises:
        ValueError: If matrix dimensions are incompatible, iteration count
            is non-positive, or the coefficient matrix is not strictly
            diagonally dominant.
    """
    _validate_inputs(coefficient_matrix, constant_matrix, initial_values, num_iterations)

    # Build augmented matrix [A | b]
    augmented_matrix = np.concatenate((coefficient_matrix, constant_matrix), axis=1)
    num_rows, num_cols = augmented_matrix.shape

    _check_strictly_diagonally_dominant(augmented_matrix)

    current_values = list(initial_values)

    for _ in range(num_iterations):
        new_values = []

        for i in range(num_rows):
            # Sum contributions from non-diagonal coefficient terms
            off_diagonal_sum = sum(
                -augmented_matrix[i][j] * current_values[j]
                for j in range(num_cols - 1)
                if j != i
            )

            # Compute updated value: (b_i - sum of off-diagonal terms) / a_ii
            diagonal_element = augmented_matrix[i][i]
            constant_term = augmented_matrix[i][num_cols - 1]
            new_values.append((off_diagonal_sum + constant_term) / diagonal_element)

        current_values = new_values

    return current_values


def _validate_inputs(
    coefficient_matrix: np.ndarray,
    constant_matrix: np.ndarray,
    initial_values: list[float],
    num_iterations: int,
) -> None:
    """
    Validate dimensions and parameters for the Jacobi iteration method.

    Raises:
        ValueError: If any input dimensions or values are invalid.
    """
    num_rows_a, num_cols_a = coefficient_matrix.shape
    num_rows_b, num_cols_b = constant_matrix.shape

    if num_rows_a != num_cols_a:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn "
            f"but received {num_rows_a}x{num_cols_a}"
        )

    if num_cols_b != 1:
        raise ValueError(
            f"Constant matrix must be nx1 "
            f"but received {num_rows_b}x{num_cols_b}"
        )

    if num_rows_a != num_rows_b:
        raise ValueError(
            f"Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {num_rows_a}x{num_cols_a} and {num_rows_b}x{num_cols_b}"
        )

    if len(initial_values) != num_rows_a:
        raise ValueError(
            f"Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(initial_values)} and {num_rows_a}"
        )

    if num_iterations <= 0:
        raise ValueError("Iterations must be at least 1")


def _check_strictly_diagonally_dominant(augmented_matrix: np.ndarray) -> None:
    """
    Verify that the coefficient portion of the augmented matrix is
    strictly diagonally dominant.

    A matrix is strictly diagonally dominant if, for every row, the
    absolute value of the diagonal element is greater than the sum
    of the absolute values of all other elements in that row.

    Args:
        augmented_matrix: The augmented matrix [A | b].

    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.
    """
    num_rows, num_cols = augmented_matrix.shape

    for i in range(num_rows):
        off_diagonal_sum = sum(
            augmented_matrix[i][j]
            for j in range(num_cols - 1)
            if j != i
        )

        if augmented_matrix[i][i] <= off_diagonal_sum:
            raise ValueError(
                "Coefficient matrix is not strictly diagonally dominant"
            )


if __name__ == "__main__":
    import doctest

    result = doctest.testmod()
    print(result)
