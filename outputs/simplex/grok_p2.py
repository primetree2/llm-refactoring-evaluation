```python
import numpy as np
from typing import Dict, List, Tuple


class Tableau:
    """Implements the two-phase simplex method using a tableau representation.
    
    Supports maximization problems with optional artificial variables for
    finding an initial basic feasible solution (Phase I).
    """

    MAX_ITERATIONS: int = 100
    """Maximum number of iterations before terminating."""

    def __init__(self, tab: np.ndarray, nv: int, na: int) -> None:
        """Initialize the simplex tableau.
        
        Args:
            tab: The initial tableau matrix (must have dtype float64).
            nv: Number of decision (structural) variables.
            na: Number of artificial variables.
            
        Raises:
            TypeError: If the tableau is not of type float64.
            ValueError: If RHS values are negative or variable counts are invalid.
        """
        if tab.dtype != "float64":
            raise TypeError("Tableau must have type float64")

        if not (tab[:, -1] >= 0).all():
            raise ValueError("RHS must be >= 0")

        if nv < 2 or na < 0:
            raise ValueError(
                "nv must be >= 2 and na must be a non-negative integer"
            )

        self.tableau: np.ndarray = tab
        self.n_rows: int = tab.shape[0]
        self.n_cols: int = tab.shape[1]

        self.n_vars: int = nv
        self.n_artificial_vars: int = na

        # Phase I (minimize artificial variables) is used when artificial vars exist
        self.n_stages: int = 2 if self.n_artificial_vars > 0 else 1
        self.n_slack: int = (
            self.n_cols - self.n_vars - self.n_artificial_vars - 1
        )

        # Objectives list drives the two-phase approach: ["max", "min"] or just ["max"]
        self.objectives: List[str] = ["max"]
        if self.n_artificial_vars > 0:
            self.objectives.append("min")

        self.col_titles: List[str] = self._generate_column_titles()

        self.stop_iter: bool = False

    def _generate_column_titles(self) -> List[str]:
        """Generate human-readable column titles for variables and RHS."""
        titles = [f"x{i+1}" for i in range(self.n_vars)]
        titles.extend([f"s{i+1}" for i in range(self.n_slack)])
        titles.append("RHS")
        return titles

    def find_pivot(self) -> Tuple[int, int]:
        """Determine the pivot position using the current objective.
        
        Uses the most negative coefficient (for max) or most positive
        (for min) in the objective row, followed by a minimum ratio test.
        
        Returns:
            Tuple of (pivot_row_index, pivot_column_index). Returns (0, 0)
            and sets stop_iter=True when the current phase is optimal.
        """
        current_objective = self.objectives[-1]
        sign = 1 if current_objective == "min" else -1

        obj_row = self.tableau[0, :-1]
        adjusted = sign * obj_row
        col_idx = int(np.argmax(adjusted))

        # No improving direction -> current phase is optimal
        if adjusted[col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        # Minimum ratio test (only consider rows after objective rows)
        start_row = self.n_stages
        rhs_values = self.tableau[start_row:, -1]
        pivot_column = self.tableau[start_row:, col_idx]

        # Avoid division by zero or negative entries
        ratios = np.divide(
            rhs_values,
            pivot_column,
            out=np.full_like(rhs_values, np.nan),
            where=(pivot_column > 0)
        )

        row_offset = int(np.nanargmin(ratios))
        row_idx = start_row + row_offset

        return row_idx, col_idx

    def pivot(self, row_idx: int, col_idx: int) -> np.ndarray:
        """Perform Gauss-Jordan elimination to pivot on the given element.
        
        Normalizes the pivot row and eliminates the pivot column in all
        other rows.
        
        Args:
            row_idx: Row index of the pivot element.
            col_idx: Column index of the pivot element.
            
        Returns:
            The updated tableau.
        """
        pivot_element = self.tableau[row_idx, col_idx]
        normalized_row = self.tableau[row_idx] / pivot_element

        # Eliminate pivot column from all other rows
        for i in range(self.tableau.shape[0]):
            if i == row_idx:
                continue
            factor = self.tableau[i, col_idx]
            self.tableau[i] -= factor * normalized_row

        self.tableau[row_idx] = normalized_row
        return self.tableau

    def change_stage(self) -> np.ndarray:
        """Transition from Phase I to Phase II.
        
        Removes the artificial variable columns and the previous objective row.
        
        Returns:
            The updated tableau for the next phase.
        """
        if not self.objectives:
            return self.tableau

        self.objectives.pop()

        if not self.objectives:
            return self.tableau

        # Remove artificial variable columns (immediately before RHS column)
        cols_to_remove = slice(-self.n_artificial_vars - 1, -1)
        self.tableau = np.delete(self.tableau, cols_to_remove, axis=1)

        # Remove the objective row from the previous phase
        self.tableau = np.delete(self.tableau, 0, axis=0)

        # Update internal state for Phase II
        self.n_stages = 1
        self.n_rows -= 1
        self.n_artificial_vars = 0
        self.stop_iter = False

        return self.tableau

    def run_simplex(self) -> Dict[str, float]:
        """Execute the full simplex algorithm (two-phase if needed).
        
        Returns:
            Dictionary containing the objective value 'P' and values of
            basic variables. Returns empty dict if maximum iterations
            are exceeded.
        """
        iteration = 0

        while iteration < Tableau.MAX_ITERATIONS:
            if not self.objectives:
                return self._interpret_solution()

            r, c = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(r, c)

            iteration += 1

        # Maximum iterations reached without convergence
        return {}

    def _interpret_solution(self) -> Dict[str, float]:
        """Extract basic variable values and objective from the final tableau.
        
        Returns:
            Dictionary with key 'P' for the objective value and keys for
            each basic decision variable.
        """
        solution: Dict[str, float] = {}
        solution["P"] = abs(self.tableau[0, -1])

        for i in range(self.n_vars):
            nonzero_rows = np.nonzero(self.tableau[:, i])[0]

            if len(nonzero_rows) == 1:
                r = nonzero_rows[0]
                coeff = self.tableau[r, i]

                # Use tolerance for floating point comparison
                if abs(coeff - 1.0) < 1e-10:
                    solution[self.col_titles[i]] = self.tableau[r, -1]

        return solution


if __name__ == "__main__":
    import doctest
    doctest.testmod()
