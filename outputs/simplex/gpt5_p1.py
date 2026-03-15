"""Simplex tableau implementation."""

from typing import Dict, List, Optional, Tuple

import numpy as np


class Tableau:
    """Represent and manipulate a simplex tableau.

    This class supports both standard simplex and two-phase simplex
    formulations, depending on whether artificial variables are present.
    """

    maxiter = 100

    def __init__(self, tab: np.ndarray, nv: int, na: int) -> None:
        """Initialize the tableau.

        Args:
            tab: Initial simplex tableau as a NumPy array of dtype float64.
            nv: Number of decision variables.
            na: Number of artificial variables.

        Raises:
            TypeError: If the tableau dtype is not float64.
            ValueError: If the RHS contains negative values or variable counts
                are invalid.
        """
        self._validate_inputs(tab, nv, na)

        self.tableau = tab
        self.n_rows, self.n_cols = tab.shape

        self.n_vars = nv
        self.n_artificial_vars = na
        self.n_stages = 2 if self.n_artificial_vars > 0 else 1
        self.n_slack = self.n_cols - self.n_vars - self.n_artificial_vars - 1

        # Objectives are processed from the end of the list.
        # If artificial variables exist, phase 1 minimizes the auxiliary
        # objective before proceeding to the original maximization problem.
        self.objectives: List[str] = ["max"]
        if self.n_artificial_vars > 0:
            self.objectives.append("min")

        self.col_titles = self._generate_titles()

        # Store the most recently selected pivot position.
        self.row_idx: Optional[int] = None
        self.col_idx: Optional[int] = None

        self.stop_iter = False

    @staticmethod
    def _validate_inputs(tab: np.ndarray, nv: int, na: int) -> None:
        """Validate constructor inputs."""
        if tab.dtype != np.dtype("float64"):
            raise TypeError("Tableau must have type float64")

        if not (tab[:, -1] >= 0).all():
            raise ValueError("RHS must be >= 0")

        if nv < 2 or na < 0:
            raise ValueError(
                "Number of variables must be >= 2 and number of artificial "
                "variables must be >= 0"
            )

    def _generate_titles(self) -> List[str]:
        """Generate column titles for decision variables, slack variables, and RHS."""
        titles = [f"x{i + 1}" for i in range(self.n_vars)]
        titles.extend(f"s{j + 1}" for j in range(self.n_slack))
        titles.append("RHS")
        return titles

    def find_pivot(self) -> Tuple[int, int]:
        """Find the next pivot position.

        Returns:
            A tuple containing the pivot row index and pivot column index.
            If the current stage is complete, returns (0, 0) and sets
            ``self.stop_iter`` to True.
        """
        objective = self.objectives[-1]
        sign = 1 if objective == "min" else -1

        # Choose the entering variable from the objective row.
        objective_row = self.tableau[0, :-1]
        col_idx = int(np.argmax(sign * objective_row))

        # If no improving pivot exists, this stage is complete.
        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            self.row_idx, self.col_idx = 0, 0
            return 0, 0

        # Apply the minimum-ratio test to choose the leaving variable.
        constraint_slice = slice(self.n_stages, self.n_rows)
        rhs = self.tableau[constraint_slice, -1]
        pivot_column = self.tableau[constraint_slice, col_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(pivot_column > 0, rhs / pivot_column, np.nan)

        row_idx = int(np.nanargmin(ratios)) + self.n_stages

        self.row_idx = row_idx
        self.col_idx = col_idx
        return row_idx, col_idx

    def pivot(self, r: int, c: int) -> np.ndarray:
        """Perform a pivot operation on the tableau.

        Args:
            r: Pivot row index.
            c: Pivot column index.

        Returns:
            The updated tableau.
        """
        pivot_row = self.tableau[r].copy()
        pivot_value = pivot_row[c]

        # Normalize the pivot row.
        pivot_row = pivot_row / pivot_value

        # Eliminate the pivot column from all rows.
        for row_index in range(self.n_rows):
            coefficient = self.tableau[row_index, c]
            self.tableau[row_index] = self.tableau[row_index] - (
                coefficient * pivot_row
            )

        # Restore the normalized pivot row explicitly.
        self.tableau[r] = pivot_row
        return self.tableau

    def change_stage(self) -> np.ndarray:
        """Transition from phase 1 to phase 2, or finish the algorithm.

        Returns:
            The updated tableau.
        """
        if self.objectives:
            self.objectives.pop()

        # No further objective remains; the next loop iteration will interpret
        # the final tableau.
        if not self.objectives:
            return self.tableau

        # Remove artificial variable columns and the auxiliary objective row.
        artificial_cols = slice(-self.n_artificial_vars - 1, -1)
        self.tableau = np.delete(self.tableau, artificial_cols, axis=1)
        self.tableau = np.delete(self.tableau, 0, axis=0)

        self.n_rows, self.n_cols = self.tableau.shape
        self.n_artificial_vars = 0
        self.n_stages = 1
        self.n_slack = self.n_cols - self.n_vars - self.n_artificial_vars - 1
        self.stop_iter = False

        return self.tableau

    def run_simplex(self) -> Dict[str, float]:
        """Run the simplex algorithm.

        Returns:
            A dictionary containing the objective value and basic variable
            assignments. Returns an empty dictionary if the iteration limit is
            reached.
        """
        for _ in range(self.maxiter):
            if not self.objectives:
                return self._interpret()

            row_idx, col_idx = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(row_idx, col_idx)

        return {}

    def _interpret(self) -> Dict[str, float]:
        """Interpret the final tableau as a solution dictionary."""
        solution: Dict[str, float] = {"P": abs(self.tableau[0, -1])}

        for col_index in range(self.n_vars):
            nonzero_rows = np.nonzero(self.tableau[:, col_index])[0]

            # A basic variable appears in exactly one row with coefficient 1.
            if len(nonzero_rows) == 1:
                row_index = nonzero_rows[0]
                if self.tableau[row_index, col_index] == 1:
                    solution[self.col_titles[col_index]] = self.tableau[row_index, -1]

        return solution


if __name__ == "__main__":
    import doctest

    doctest.testmod()
