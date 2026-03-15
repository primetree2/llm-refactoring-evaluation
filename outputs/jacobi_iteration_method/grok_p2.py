"""
Module providing the Jacobi iteration method for solving linear equations.
"""

import numpy as np


def jacobi_iteration_method(
    a: np.ndarray, b: np.ndarray, x0: list[float], iters: int
) -> list[float]:
    """
    Solve a system of linear equations using the Jacobi iteration method.

    The method expects a strictly diagonally dominant coefficient matrix `a`
    and a constant matrix `b`, along with an initial guess `x0`.

    Args:
        a: Coefficient matrix of shape (n, n).
        b: Constant matrix of shape (n, 1).
        x0: Initial guess for the solution, length n.
        iters: Number of iterations to perform.

    Returns:
        The approximated solution vector after the specified iterations.

    Raises:
        ValueError: If matrix dimensions are incompatible, if the number of
            initial values does not match, if iterations are <= 0, or if the
            coefficient matrix is not strictly diagonally dominant.
    """
    _validate_inputs(a, b, x0, iters)

    # Build the augmented matrix [A | b]
    table = np.concatenate((a, b), axis=1)
    rows, cols = table.shape

    _check_strictly_diagonally_dominant(table)

    current_values = list(x0)

    for _ in range(iters):
        new_vals = []

        for i in range(rows):
            # Sum (-a_ij * x_j) for all j != i
            off_diagonal_sum = sum(
                -table[i][j] * current_values[j]
                for j in range(cols - 1)
                if j != i
            )
            
            diagonal_element = table[i][i]
            constant_term = table[i][cols - 1]
            
            # x_i = (sum_{j!=i} (-a_ij * x_j) + b_i) / a_ii
            value = (off_diagonal_sum + constant_term) / diagonal_element
            new_vals.append(value)

        current_values = new_vals

    return current_values


def _validate_inputs(
    a: np.ndarray, b: np.ndarray, x0: list[float], iters: int
) -> None:
    """
    Validate the inputs to the Jacobi iteration method.

    Raises:
        ValueError: If input dimensions are invalid.
    """
    r1, c1 = a.shape
    r2, c2 = b.shape

    if r1 != c1:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received {r1}x{c1}"
        )

    if c2 != 1:
        raise ValueError(
            f"Constant matrix must be nx1 but received {r2}x{c2}"
        )

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


def _check_strictly_diagonally_dominant(table: np.ndarray) -> None:
    """
    Check if the coefficient part of the augmented matrix is strictly
    diagonally dominant.

    Note: 
        This preserves the original logic, which does not use absolute values 
        when summing off-diagonal elements.

    Args:
        table: Augmented matrix of shape (n, n + 1).

    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.
    """
    rows, cols = table.shape

    for i in range(rows):
        off_diagonal_sum = sum(
            table[i][j] for j in range(cols - 1) if i != j
        )

        if table[i][i] <= off_diagonal_sum:
            raise ValueError(
                "Coefficient matrix is not strictly diagonally dominant"
            )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
