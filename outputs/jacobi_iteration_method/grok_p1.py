"""
Jacobi iteration method for solving systems of linear equations.
Requires the coefficient matrix to be strictly diagonally dominant.
"""

import numpy as np
from typing import List, Union


def check_strict_diagonal_dominance(a: np.ndarray) -> bool:
    """
    Verify that the matrix is strictly diagonally dominant.

    Raises:
        ValueError: If the matrix is not strictly diagonally dominant.

    Returns:
        True if the check passes.
    """
    rows, cols = a.shape[0], a.shape[1]

    for i in range(rows):
        off_diagonal_sum = 0.0
        for j in range(cols):
            if i != j:
                off_diagonal_sum += a[i, j]

        if a[i, i] <= off_diagonal_sum:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return True


def jacobi_iteration_method(
    a: np.ndarray,
    b: np.ndarray,
    x0: Union[List[float], np.ndarray],
    max_iters: int,
) -> List[float]:
    """
    Solve the linear system Ax = b using the Jacobi iterative method.

    The coefficient matrix must be strictly diagonally dominant for
    guaranteed convergence.

    Args:
        a: Coefficient matrix (n x n).
        b: Constant vector (n x 1).
        x0: Initial guess vector.
        max_iters: Maximum number of iterations to perform.

    Returns:
        The approximated solution vector after max_iters iterations.

    Raises:
        ValueError: For invalid matrix dimensions, incompatible sizes,
                    or non-diagonally dominant coefficient matrix.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received "
            f"{a.shape[0]}x{a.shape[1]}"
        )

    if b.ndim != 2 or b.shape[1] != 1:
        raise ValueError(
            f"Constant matrix must be nx1 but received "
            f"{b.shape[0]}x{b.shape[1]}"
        )

    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError(
            f"Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {a.shape} and {b.shape}"
        )

    if len(x0) != n:
        raise ValueError(
            f"Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(x0)} and {n}"
        )

    if max_iters <= 0:
        raise ValueError("Iterations must be at least 1")

    # Ensure we work with a numpy array for the coefficient matrix
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = list(x0)  # work with Python list to preserve original behavior

    check_strict_diagonal_dominance(a)

    for _ in range(max_iters):
        new_x: List[float] = []

        for i in range(n):
            total = 0.0
            for j in range(n):
                if j != i:
                    total -= a[i, j] * x[j]

            value = (total + b[i, 0]) / a[i, i]
            new_x.append(value)

        x = new_x

    return x


if __name__ == "__main__":
    import doctest

    # Run any embedded doctests (none currently present)
    result = doctest.testmod()
    print(result)
