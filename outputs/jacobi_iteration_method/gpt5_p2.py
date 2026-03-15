"""Jacobi iteration method for solving systems of linear equations."""

from __future__ import annotations

import numpy as np


def jacobi_iteration_method(
    a: np.ndarray,
    b: np.ndarray,
    x0: list[float],
    iters: int,
) -> list[float]:
    """
    Approximate the solution of a linear system using the Jacobi iteration method.

    Args:
        a: Coefficient matrix of shape (n, n).
        b: Constant matrix of shape (n, 1).
        x0: Initial guess for the solution vector.
        iters: Number of iterations to perform.

    Returns:
        A list containing the approximated solution after the specified
        number of iterations.

    Raises:
        ValueError: If the input dimensions are incompatible, the number of
            initial values does not match the system size, the iteration count
            is invalid, or the coefficient matrix is not strictly diagonally
            dominant.
    """
    _validate_inputs(a, b, x0, iters)

    # Build the augmented matrix [A | b] to preserve the original approach.
    table = np.concatenate((a, b), axis=1)
    rows, cols = table.shape

    # Validate the convergence condition before iterating.
    strictly_diagonally_dominant(table)

    # Copy the initial values to avoid mutating the input sequence.
    current = list(x0)

    for _ in range(iters):
        new_vals: list[float] = []

        for i in range(rows):
            total = 0
            denom = None
            last_val = None

            for j in range(cols):
                if j == i:
                    denom = table[i][j]
                elif j == cols - 1:
                    last_val = table[i][j]
                else:
                    total += (-1) * table[i][j] * current[j]

            new_vals.append((total + last_val) / denom)

        current = list(new_vals)

    return current


def strictly_diagonally_dominant(table: np.ndarray) -> bool:
    """
    Check whether the coefficient portion of an augmented matrix is strictly
    diagonally dominant.

    Note:
        This function preserves the original comparison logic exactly. In
        particular, it sums non-diagonal entries directly rather than using
        absolute values.

    Args:
        table: Augmented matrix of shape (n, n + 1).

    Returns:
        True if the matrix passes the strict diagonal dominance check.

    Raises:
        ValueError: If the coefficient matrix is not strictly diagonally dominant.
    """
    rows, cols = table.shape

    for i in range(rows):
        row_sum = 0

        for j in range(cols - 1):
            if i == j:
                continue
            row_sum += table[i][j]

        if table[i][i] <= row_sum:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return True


def _validate_inputs(
    a: np.ndarray,
    b: np.ndarray,
    x0: list[float],
    iters: int,
) -> None:
    """
    Validate the inputs for the Jacobi iteration method.

    Args:
        a: Coefficient matrix.
        b: Constant matrix.
        x0: Initial guess values.
        iters: Number of iterations.

    Raises:
        ValueError: If any input does not satisfy the expected dimensions
            or constraints.
    """
    r1, c1 = a.shape
    r2, c2 = b.shape

    if r1 != c1:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received {r1}x{c1}"
        )

    if c2 != 1:
        raise ValueError(f"Constant matrix must be nx1 but received {r2}x{c2}")

    if r1 != r2:
        raise ValueError(
            "Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {r1}x{c1} and {r2}x{c2}"
        )

    if len(x0) != r1:
        raise ValueError(
            "Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(x0)} and {r1}"
        )

    if iters <= 0:
        raise ValueError("Iterations must be at least 1")


if __name__ == "__main__":
    import doctest

    result = doctest.testmod()
    print(result)
