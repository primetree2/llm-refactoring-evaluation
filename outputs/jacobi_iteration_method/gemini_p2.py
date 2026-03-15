"""
Module for solving systems of linear equations using the Jacobi iteration method.
"""

import numpy as np


def jacobi_iteration_method(
    coeff_matrix: np.ndarray,
    const_matrix: np.ndarray,
    initial_guess: list[float],
    iterations: int,
) -> list[float]:
    """
    Approximates the solution to a system of linear equations (Ax = b) 
    using the Jacobi iteration method.

    Args:
        coeff_matrix (np.ndarray): An n x n array of coefficients (A).
        const_matrix (np.ndarray): An n x 1 array of constants (b).
        initial_guess (list[float]): A list of length n with initial guess values.
        iterations (int): The number of iterations to perform.

    Returns:
        list[float]: The approximated solution vector after the specified iterations.

    Raises:
        ValueError: If matrix dimensions are incompatible, the iteration count 
            is invalid, or the coefficient matrix is not strictly diagonally dominant.
    """
    _validate_inputs(coeff_matrix, const_matrix, initial_guess, iterations)

    # Build the augmented matrix [A | b]
    augmented_matrix = np.concatenate((coeff_matrix, const_matrix), axis=1)
    
    _check_strict_diagonal_dominance(augmented_matrix)

    # Copy the initial guess to avoid mutating the original input list
    current_solution = list(initial_guess)
    num_rows, num_cols = augmented_matrix.shape

    # Perform the Jacobi iterations
    for _ in range(iterations):
        next_solution = []

        for i in range(num_rows):
            # Sum the contributions from all non-diagonal elements: sum(-a_ij * x_j)
            off_diagonal_sum = sum(
                -augmented_matrix[i, j] * current_solution[j]
                for j in range(num_cols - 1)
                if j != i
            )
            
            # x_i = (off_diagonal_sum + b_i) / a_ii
            diagonal_element = augmented_matrix[i, i]
            constant_term = augmented_matrix[i, num_cols - 1]
            
            new_value = (off_diagonal_sum + constant_term) / diagonal_element
            next_solution.append(new_value)

        current_solution = next_solution

    return current_solution


def _validate_inputs(
    coeff_matrix: np.ndarray,
    const_matrix: np.ndarray,
    initial_guess: list[float],
    iterations: int,
) -> None:
    """
    Validates the input parameters for the Jacobi iteration method.

    Raises:
        ValueError: If matrix dimensions are incorrect or iterations < 1.
    """
    rows_a, cols_a = coeff_matrix.shape
    rows_b, cols_b = const_matrix.shape

    if rows_a != cols_a:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received {rows_a}x{cols_a}"
        )

    if cols_b != 1:
        raise ValueError(
            f"Constant matrix must be nx1 but received {rows_b}x{cols_b}"
        )

    if rows_a != rows_b:
        raise ValueError(
            "Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {rows_a}x{cols_a} and {rows_b}x{cols_b}"
        )

    if len(initial_guess) != rows_a:
        raise ValueError(
            "Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(initial_guess)} and {rows_a}"
        )

    if iterations <= 0:
        raise ValueError("Iterations must be at least 1")


def _check_strict_diagonal_dominance(augmented_matrix: np.ndarray) -> None:
    """
    Checks if the coefficient portion of the augmented matrix is strictly 
    diagonally dominant.

    Note:
        This preserves the original logic which compares against the sum 
        of the raw (non-absolute) values of the off-diagonal elements.

    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.
    """
    rows, cols = augmented_matrix.shape

    for i in range(rows):
        off_diagonal_sum = sum(
            augmented_matrix[i, j]
            for j in range(cols - 1)
            if j != i
        )

        if augmented_matrix[i, i] <= off_diagonal_sum:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
