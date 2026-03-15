"""Simplex tableau implementation for linear programming."""

from typing import Dict, List, Tuple

import numpy as np


class Tableau:
    """Simplex tableau supporting one- and two-phase methods."""

    maxiter = 100

    def __init__(self, tab: np.ndarray, nv: int, na: int) -> None:
        """Initialize a new simplex tableau.

        Args:
            tab: Initial tableau (must be of dtype float64).
            nv: Number of decision (original) variables.
            na: Number of artificial variables.

        Raises:
            TypeError: If tableau dtype is not float64.
            ValueError: If RHS is negative or variable counts are invalid.
        """
        self._validate_inputs(tab, nv, na)

        self.tableau = tab
        self.n_rows, self.n_cols = tab.shape

        self.n_vars = nv
        self.n_artificial_vars = na
        self.n_stages = 2 if na > 0 else 1
        self.n_slack = self.n_cols - self.n_vars - self.n_artificial_vars - 1

        self.objectives: List[str] = ["max"]
        if self.n_artificial_vars > 0:
            self.objectives.append("min")

        self.col_titles = self._generate_column_titles()

        self.row_idx = None
        self.col_idx = None
        self.stop_iter = False

    @staticmethod
    def _validate_inputs(tab: np.ndarray, nv: int, na: int) -> None:
        """Validate constructor arguments."""
        if tab.dtype != np.float64:
            raise TypeError("Tableau must have type float64")

        if not (tab[:, -1] >= 0).all():
            raise ValueError("RHS must be >= 0")

        if nv < 2 or na < 0:
            raise ValueError(
                "number of (artificial) variables must be a natural number"
            )

    def _generate_column_titles(self) -> List[str]:
        """Generate column header names for variables and RHS."""
        titles = [f"x{i + 1}" for i in range(self.n_vars)]
        titles.extend(f"s{j + 1}" for j in range(self.n_slack))
        titles.append("RHS")
        return titles

    def find_pivot(self) -> Tuple[int, int]:
        """Determine the pivot element using the current objective.

        Returns:
            Tuple of (pivot_row, pivot_col). Returns (0, 0) and flags
            stop_iter when the current stage has reached optimality.
        """
        objective = self.objectives[-1]
        sign = 1 if objective == "min" else -1

        objective_row = self.tableau[0, :-1]
        col_idx = int(np.argmax(sign * objective_row))

        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        constraint_slice = slice(self.n_stages, self.n_rows)
        rhs = self.tableau[constraint_slice, -1]
        col = self.tableau[constraint_slice, col_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(col > 0, rhs / col, np.nan)

        row_idx = int(np.nanargmin(ratios)) + self.n_stages
        return row_idx, col_idx

    def pivot(self, r: int, c: int) -> np.ndarray:
        """Execute the pivot operation on the tableau.

        Args:
            r: Pivot row index.
            c: Pivot column index.

        Returns:
            Updated tableau.
        """
        pivot_row = self.tableau[r].copy()
        pivot_val = pivot_row[c]
        normalized = pivot_row / pivot_val

        for i in range(self.n_rows):
            coeff = self.tableau[i, c]
            self.tableau[i] = self.tableau[i] - coeff * normalized

        self.tableau[r] = normalized
        return self.tableau

    def change_stage(self) -> np.ndarray:
        """Remove artificial variables and auxiliary objective row.

        Returns:
            Updated tableau for the next stage.
        """
        if self.objectives:
            self.objectives.pop()

        if not self.objectives:
            return self.tableau

        # Remove artificial variable columns
        artificial_slice = slice(-self.n_artificial_vars - 1, -1)
        self.tableau = np.delete(self.tableau, artificial_slice, axis=1)
        # Remove auxiliary objective row
        self.tableau = np.delete(self.tableau, 0, axis=0)

        self.n_rows, self.n_cols = self.tableau.shape
        self.n_stages = 1
        self.n_artificial_vars = 0
        self.n_slack = self.n_cols - self.n_vars - 1
        self.stop_iter = False

        return self.tableau

    def run_simplex(self) -> Dict[str, float]:
        """Run the full simplex algorithm.

        Returns:
            Dictionary with objective value "P" and basic variable values,
            or empty dict if maximum iterations are exceeded.
        """
        iteration = 0
        while iteration < self.maxiter:
            if not self.objectives:
                return self._interpret()

            r, c = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(r, c)

            iteration += 1

        return {}

    def _interpret(self) -> Dict[str, float]:
        """Extract solution values from the final tableau.

        Returns:
            Dictionary containing the objective and basic variable values.
        """
        solution: Dict[str, float] = {"P": abs(self.tableau[0, -1])}

        for i in range(self.n_vars):
            nz_rows = np.nonzero(self.tableau[:, i])[0]

            if len(nz_rows) == 1:
                row = nz_rows[0]
                if self.tableau[row, i] == 1.0:
                    solution[self.col_titles[i]] = self.tableau[row, -1]

        return solution


if __name__ == "__main__":
    import doctest

    doctest.testmod()
