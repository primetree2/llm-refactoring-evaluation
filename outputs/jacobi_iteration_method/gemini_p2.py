"""
Module for solving systems of linear equations using the Jacobi iteration method.
"""

import numpy as np


def jacobi_iteration_method(
    a: np.ndarray, 
    b: np.ndarray, 
    x0: list[float], 
    iters: int
) -> list[float]:
    """
    Solves a system of linear equations using the Jacobi iteration method.

    Args:
        a (np.ndarray): Coefficient matrix of shape (n, n).
        b (np.ndarray): Constant matrix of shape (n, 1).
        x0 (list[float]): Initial guess vector of length n.
        iters (int): Number of iterations to perform.

    Returns:
        list[float]: The approximated solution vector.

    Raises:
        ValueError: If matrix dimensions are incorrect, iterations are invalid,
                    or the coefficient matrix is not strictly diagonally dominant.
    """
    rows_a, cols_a = a.shape
    rows_b, cols_b = b.shape

    # Validate dimensions and inputs
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
            f"Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {rows_a}x{cols_a} and {rows_b}x{cols_b}"
        )

    if len(x0) != rows_a:
        raise ValueError(
            f"Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(x0)} and {rows_a}"
        )

    if iters <= 0:
        raise ValueError("Iterations must be at least 1")

    # Check for diagonal dominance to ensure convergence
    strictly_diagonally_dominant(a)

    # Initialize current values with a copy of x0
    current = list(x0)

    # Main iteration loop
    for _ in range(iters):
        new_vals = []
        
        for i in range(rows_a):
            # Sum contributions from all off-diagonal elements
            off_diag_sum = sum(a[i, j] * current[j] for j in range(cols_a) if j != i)
            
            # Calculate the new value for the i-th variable
            value = (b[i, 0] - off_diag_sum) / a[i, i]
            new_vals.append(value)

        current = new_vals

    return current


def strictly_diagonally_dominant(matrix: np.ndarray) -> bool:
    """
    Checks if the given coefficient matrix is strictly diagonally dominant.
    
    Note: 
        This implementation preserves the original logic, which does not 
        use absolute values for the sum of off-diagonal elements.

    Args:
        matrix (np.ndarray): The coefficient matrix to evaluate.

    Returns:
        bool: True if the matrix is strictly diagonally dominant.

    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.
    """
    rows, cols = matrix.shape

    for i in range(rows):
        off_diag_sum = sum(matrix[i, j] for j in range(cols) if j != i)

        if matrix[i, i] <= off_diag_sum:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
