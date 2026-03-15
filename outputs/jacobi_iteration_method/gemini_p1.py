"""
Module for solving systems of linear equations using the Jacobi iteration method.
"""

import numpy as np


def _validate_inputs(a: np.ndarray, b: np.ndarray, x0: list[float], iters: int) -> None:
    """
    Validates the dimensions and values of the inputs for the Jacobi method.
    
    Args:
        a: Coefficient matrix.
        b: Constant matrix.
        x0: Initial guess list.
        iters: Number of iterations.
        
    Raises:
        ValueError: If matrix dimensions are incorrect or iterations < 1.
    """
    r1, c1 = a.shape
    r2, c2 = b.shape

    if r1 != c1:
        raise ValueError(f"Coefficient matrix dimensions must be nxn but received {r1}x{c1}")

    if c2 != 1:
        raise ValueError(f"Constant matrix must be nx1 but received {r2}x{c2}")

    if r1 != r2:
        raise ValueError(
            f"Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {r1}x{c1} and {r2}x{c2}"
        )

    if len(x0) != r1:
        raise ValueError(
            f"Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(x0)} and {r1}"
        )

    if iters <= 0:
        raise ValueError("Iterations must be at least 1")


def check_strictly_diagonally_dominant(coeff_matrix: np.ndarray) -> bool:
    """
    Checks if the coefficient matrix is strictly diagonally dominant.
    
    Note: 
        This implementation preserves the original logic which does not 
        use absolute values for the sum of off-diagonal elements.
        
    Args:
        coeff_matrix: The nxn coefficient matrix.
        
    Returns:
        True if the matrix satisfies the condition.
        
    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.
    """
    rows, cols = coeff_matrix.shape

    for i in range(rows):
        off_diag_sum = sum(coeff_matrix[i, j] for j in range(cols) if j != i)
        
        if coeff_matrix[i, i] <= off_diag_sum:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return True


def jacobi_iteration_method(a: np.ndarray, b: np.ndarray, x0: list[float], iters: int) -> list[float]:
    """
    Solves a system of linear equations (Ax = b) using the Jacobi iteration method.
    
    Args:
        a: An nxn numpy array representing the coefficient matrix (A).
        b: An nx1 numpy array representing the constant matrix (b).
        x0: A list of length n representing the initial guesses.
        iters: The number of iterations to perform.
        
    Returns:
        A list containing the approximated solution values after the specified iterations.
    """
    # Validate matrices and initial parameters
    _validate_inputs(a, b, x0, iters)
    
    # Check for diagonal dominance to ensure convergence
    check_strictly_diagonally_dominant(a)

    # Initialize current values with a copy of the initial guesses
    current = list(x0)
    rows = a.shape[0]

    # Perform Jacobi iterations
    for _ in range(iters):
        new_vals = []
        
        for i in range(rows):
            # Sum (-a_ij * x_j) for all j != i
            off_diag_sum = sum(-a[i, j] * current[j] for j in range(rows) if j != i)
            
            # x_i = (sum(-a_ij * x_j) + b_i) / a_ii
            value = (off_diag_sum + b[i, 0]) / a[i, i]
            new_vals.append(value)

        current = new_vals

    return current


if __name__ == "__main__":
    import doctest
    doctest.testmod()
