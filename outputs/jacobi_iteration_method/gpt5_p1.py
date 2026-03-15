"""Jacobi iteration method implementation for solving linear systems."""

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
        x0: Initial guess values.
        iters: Number of Jacobi iterations to perform.

    Returns:
        A list containing the approximated solution values after the
        specified number of iterations.

    Raises:
        ValueError: If the input dimensions are invalid, if the number of
            initial values does not match the matrix size, if the number of
            iterations is less than 1, or if the coefficient matrix is not
            strictly diagonally dominant.
    """
    _validate_inputs(a, b, x0, iters)

    # Build the augmented matrix [A | b] to preserve the original algorithm.
    table = np.concatenate((a, b), axis=1)
    rows, cols = table.shape

    # Validate the matrix before iterating.
    strictly_diagonally_dominant(table)

    # Copy initial values so the original input is not modified.
    current_values = list(x0)

    for _ in range(iters):
        next_values: list[float] = []

        for row_index in range(rows):
            total = 0
            diagonal_value = None
            constant_value = None

            for col_index in range(cols):
                if col_index == row_index:
                    diagonal_value = table[row_index][col_index]
                elif col_index == cols - 1:
                    constant_value = table[row_index][col_index]
                else:
                    total += (-1) * table[row_index][col_index] * current_values[col_index]

            next_value = (total + constant_value) / diagonal_value
            next_values.append(next_value)

        current_values = list(next_values)

    return current_values


def strictly_diagonally_dominant(table: np.ndarray) -> bool:
    """
    Check whether the coefficient portion of an augmented matrix is strictly
    diagonally dominant.

    Note:
        This function preserves the original comparison logic exactly and does
        not use absolute values when summing non-diagonal elements.

    Args:
        table: Augmented matrix of shape (n, n + 1).

    Returns:
        True if the matrix passes the strict diagonal dominance check.

    Raises:
        ValueError: If the coefficient matrix is not strictly diagonally dominant.
    """
    rows, cols = table.shape

    for row_index in range(rows):
        non_diagonal_sum = 0

        for col_index in range(cols - 1):
            if row_index == col_index:
                continue
            non_diagonal_sum += table[row_index][col_index]

        if table[row_index][row_index] <= non_diagonal_sum:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

    return True


def _validate_inputs(
    a: np.ndarray,
    b: np.ndarray,
    x0: list[float],
    iters: int,
) -> None:
    """
    Validate inputs for the Jacobi iteration method.

    Args:
        a: Coefficient matrix.
        b: Constant matrix.
        x0: Initial guess values.
        iters: Number of iterations.

    Raises:
        ValueError: If any input is incompatible with the algorithm.
    """
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    if a_rows != a_cols:
        raise ValueError(
            f"Coefficient matrix dimensions must be nxn but received {a_rows}x{a_cols}"
        )

    if b_cols != 1:
        raise ValueError(
            f"Constant matrix must be nx1 but received {b_rows}x{b_cols}"
        )

    if a_rows != b_rows:
        raise ValueError(
            "Coefficient and constant matrices dimensions must be nxn and nx1 "
            f"but received {a_rows}x{a_cols} and {b_rows}x{b_cols}"
        )

    if len(x0) != a_rows:
        raise ValueError(
            "Number of initial values must be equal to number of rows in "
            f"coefficient matrix but received {len(x0)} and {a_rows}"
        )

    if iters <= 0:
        raise ValueError("Iterations must be at least 1")


if __name__ == "__main__":
    import doctest

    result = doctest.testmod()
    print(result)
