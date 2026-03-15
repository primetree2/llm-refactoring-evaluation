"""Two-phase simplex method implementation using a tableau representation."""

import numpy as np
from typing import Dict, List, Tuple


class Tableau:
    """Solves linear programming problems using the two-phase simplex method.

    The tableau encapsulates the LP problem data and provides methods to
    iteratively pivot toward an optimal solution. Phase I minimizes
    artificial variables to find a basic feasible solution; Phase II
    maximizes (or minimizes) the original objective.

    Attributes:
        MAX_ITERATIONS: Upper bound on simplex iterations to prevent cycling.
    """

    MAX_ITERATIONS: int = 100

    def __init__(self, tab: np.ndarray, num_vars: int, num_artificial: int) -> None:
        """Initialize the Tableau with validated inputs.

        Args:
            tab: A 2-D float64 NumPy array representing the initial tableau.
                 The last column is the RHS, and all RHS entries must be >= 0.
            num_vars: Number of decision variables (must be >= 2).
            num_artificial: Number of artificial variables (must be >= 0).

        Raises:
            TypeError: If ``tab`` does not have dtype float64.
            ValueError: If any RHS value is negative or variable counts are invalid.
        """
        if tab.dtype != np.float64:
            raise TypeError("Tableau must have dtype float64.")

        if not (tab[:, -1] >= 0).all():
            raise ValueError("All RHS values must be >= 0.")

        if num_vars < 2 or num_artificial < 0:
            raise ValueError(
                "num_vars must be >= 2 and num_artificial must be >= 0."
            )

        self.tableau: np.ndarray = tab
        self.n_rows: int = tab.shape[0]
        self.n_cols: int = tab.shape[1]
        self.n_vars: int = num_vars
        self.n_artificial_vars: int = num_artificial

        # Two-phase method is required when artificial variables are present.
        self.n_stages: int = 2 if self.n_artificial_vars > 0 else 1

        # Slack variables fill the columns between decision and artificial vars.
        self.n_slack: int = self.n_cols - self.n_vars - self.n_artificial_vars - 1

        # Build the objective stack: Phase II ("max") is always present;
        # Phase I ("min") is added on top when artificial variables exist.
        self.objectives: List[str] = ["max"]
        if self.n_artificial_vars > 0:
            self.objectives.append("min")

        self.col_titles: List[str] = self._generate_column_titles()
        self.stop_iter: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_column_titles(self) -> List[str]:
        """Create human-readable column labels (x1, x2, …, s1, s2, …, RHS).

        Returns:
            Ordered list of column title strings.
        """
        titles = [f"x{i + 1}" for i in range(self.n_vars)]
        titles += [f"s{j + 1}" for j in range(self.n_slack)]
        titles.append("RHS")
        return titles

    # ------------------------------------------------------------------
    # Pivot selection and execution
    # ------------------------------------------------------------------

    def find_pivot(self) -> Tuple[int, int]:
        """Identify the pivot element for the current simplex iteration.

        For a *maximization* objective the most negative coefficient in the
        objective row is selected (Dantzig's rule).  For *minimization* the
        most positive coefficient is chosen.  The pivot row is then determined
        by the minimum-ratio test.

        Returns:
            A ``(row_index, col_index)`` tuple.  If the current phase is
            already optimal, ``stop_iter`` is set to ``True`` and ``(0, 0)``
            is returned.
        """
        current_objective = self.objectives[-1]
        sign = 1 if current_objective == "min" else -1

        # Select the entering variable (pivot column).
        objective_row = self.tableau[0, :-1]
        col_idx = int(np.argmax(sign * objective_row))

        # Optimality check: no improving direction exists.
        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        # Minimum-ratio test to select the leaving variable (pivot row).
        constraint_rows = slice(self.n_stages, self.n_rows)
        rhs = self.tableau[constraint_rows, -1]
        column = self.tableau[constraint_rows, col_idx]

        ratios = np.where(column > 0, rhs / column, np.nan)
        row_idx = int(np.nanargmin(ratios)) + self.n_stages

        return row_idx, col_idx

    def pivot(self, pivot_row: int, pivot_col: int) -> np.ndarray:
        """Perform a single pivot (Gauss-Jordan elimination step).

        The pivot row is scaled so that the pivot element becomes 1, then
        every other row is updated to zero-out its entry in the pivot column.

        Args:
            pivot_row: Row index of the pivot element.
            pivot_col: Column index of the pivot element.

        Returns:
            The updated tableau array.
        """
        # Normalize the pivot row.
        pivot_value = self.tableau[pivot_row, pivot_col]
        normalized_row = self.tableau[pivot_row] / pivot_value

        # Eliminate the pivot column in all other rows.
        for i in range(self.n_rows):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i] -= factor * normalized_row

        self.tableau[pivot_row] = normalized_row
        return self.tableau

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def change_stage(self) -> np.ndarray:
        """Transition from Phase I to Phase II.

        Removes the artificial-variable columns and the Phase I objective
        row, then resets internal counters for a single-stage solve.

        Returns:
            The updated tableau array ready for Phase II.
        """
        if self.objectives:
            self.objectives.pop()

        if not self.objectives:
            return self.tableau

        # Remove artificial-variable columns (located just before the RHS).
        art_cols = slice(-self.n_artificial_vars - 1, -1)
        self.tableau = np.delete(self.tableau, art_cols, axis=1)

        # Remove the Phase I objective row.
        self.tableau = np.delete(self.tableau, 0, axis=0)

        # Update bookkeeping for Phase II.
        self.n_stages = 1
        self.n_rows -= 1
        self.n_artificial_vars = 0
        self.stop_iter = False

        return self.tableau

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------

    def run_simplex(self) -> Dict[str, float]:
        """Execute the simplex algorithm across all phases.

        Returns:
            A dictionary mapping ``'P'`` to the optimal objective value and
            each basic decision variable (e.g. ``'x1'``) to its value.
            Returns an empty dictionary if the iteration limit is reached.
        """
        for _ in range(self.MAX_ITERATIONS):
            # All phases complete – extract the solution.
            if not self.objectives:
                return self._interpret_solution()

            pivot_row, pivot_col = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(pivot_row, pivot_col)

        # Iteration limit exceeded without finding an optimal solution.
        return {}

    def _interpret_solution(self) -> Dict[str, float]:
        """Read basic-variable values from the final tableau.

        A decision variable is basic if its column contains exactly one
        non-zero entry equal to 1; the corresponding RHS value is then
        reported.

        Returns:
            Solution dictionary with the objective value under key ``'P'``
            and each basic decision variable's value under its title.
        """
        solution: Dict[str, float] = {"P": abs(self.tableau[0, -1])}

        for i in range(self.n_vars):
            nonzero_rows = np.nonzero(self.tableau[:, i])[0]

            # Column must have exactly one non-zero entry equal to 1.
            if len(nonzero_rows) == 1:
                row = nonzero_rows[0]
                if np.isclose(self.tableau[row, i], 1.0):
                    solution[self.col_titles[i]] = self.tableau[row, -1]

        return solution


if __name__ == "__main__":
    import doctest

    doctest.testmod()
