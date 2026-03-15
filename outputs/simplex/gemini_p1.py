"""Simplex method implementation using tableau operations."""

from typing import Any, Dict, List, Tuple
import numpy as np


class Tableau:
    """Represents a simplex tableau for solving linear programming problems.

    The tableau supports both single-stage and two-stage simplex methods,
    handling artificial variables when necessary.

    Attributes:
        MAX_ITERATIONS (int): Maximum number of simplex iterations allowed.
    """

    MAX_ITERATIONS = 100

    def __init__(self, tab: np.ndarray, n_vars: int, n_artificial_vars: int) -> None:
        """Initialize the simplex tableau.

        Args:
            tab (np.ndarray): A 2D numpy array of dtype float64 representing the initial tableau.
                              The last column is the right-hand side (RHS), which must be non-negative.
            n_vars (int): Number of decision variables (must be >= 2).
            n_artificial_vars (int): Number of artificial variables (must be >= 0).

        Raises:
            TypeError: If the tableau does not have dtype float64.
            ValueError: If the RHS contains negative values or variable counts are invalid.
        """
        self._validate_inputs(tab, n_vars, n_artificial_vars)

        self.tableau = tab
        self.n_rows, self.n_cols = tab.shape

        self.n_vars = n_vars
        self.n_artificial_vars = n_artificial_vars

        # Two stages are required if artificial variables are present
        self.n_stages = 2 if self.n_artificial_vars > 0 else 1
        self.n_slack = self.n_cols - self.n_vars - self.n_artificial_vars - 1

        # Stage objectives: "max" for the primary, "min" for the artificial stage.
        # Objectives are processed in reverse order (last objective is current).
        self.objectives = ["max"]
        if self.n_artificial_vars > 0:
            self.objectives.append("min")

        self.col_titles = self._generate_column_titles()
        self.stop_iter = False

    @staticmethod
    def _validate_inputs(tab: np.ndarray, n_vars: int, n_artificial_vars: int) -> None:
        """Validate tableau inputs.

        Args:
            tab (np.ndarray): The tableau array to validate.
            n_vars (int): Number of decision variables.
            n_artificial_vars (int): Number of artificial variables.

        Raises:
            TypeError: If the tableau dtype is not float64.
            ValueError: If RHS values are negative or variable counts are invalid.
        """
        if tab.dtype != np.float64:
            raise TypeError("Tableau must have type float64")
        if not (tab[:, -1] >= 0).all():
            raise ValueError("RHS must be >= 0")
        if n_vars < 2 or n_artificial_vars < 0:
            raise ValueError(
                "Number of variables must be >= 2 and number of artificial variables must be >= 0"
            )

    def _generate_column_titles(self) -> List[str]:
        """Generate human-readable titles for each column of the tableau.

        Returns:
            List[str]: A list of column title strings (e.g., ["x1", "x2", "s1", "RHS"]).
        """
        titles = [f"x{i + 1}" for i in range(self.n_vars)]
        titles += [f"s{j + 1}" for j in range(self.n_slack)]
        titles.append("RHS")
        return titles

    def find_pivot(self) -> Tuple[int, int]:
        """Find the pivot row and column for the next simplex iteration.

        Uses the largest-coefficient rule for column selection and the
        minimum-ratio test for row selection.

        Returns:
            Tuple[int, int]: A tuple (row_index, col_index) of the pivot element.
                             Returns (0, 0) and sets `stop_iter` to True if optimal.
        """
        current_objective = self.objectives[-1]
        sign = 1 if current_objective == "min" else -1

        # Select pivot column: the variable with the largest scaled coefficient
        objective_row = self.tableau[0, :-1]
        col_idx = int(np.argmax(sign * objective_row))

        # Check optimality condition
        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        # Select pivot row using minimum-ratio test (only constraint rows)
        constraint_rows = slice(self.n_stages, self.n_rows)
        rhs = self.tableau[constraint_rows, -1]
        column = self.tableau[constraint_rows, col_idx]

        # Compute ratios; use NaN for non-positive column entries
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(column > 0, rhs / column, np.nan)

        row_idx = int(np.nanargmin(ratios)) + self.n_stages
        return row_idx, col_idx

    def pivot(self, pivot_row: int, pivot_col: int) -> np.ndarray:
        """Perform a pivot operation on the tableau.

        Normalizes the pivot row and eliminates the pivot column entry
        from all other rows.

        Args:
            pivot_row (int): Row index of the pivot element.
            pivot_col (int): Column index of the pivot element.

        Returns:
            np.ndarray: The updated tableau after pivoting.
        """
        pivot_value = self.tableau[pivot_row, pivot_col]
        normalized_row = self.tableau[pivot_row] / pivot_value

        # Eliminate the pivot column from all other rows
        for i in range(self.n_rows):
            if i != pivot_row:
                coefficient = self.tableau[i, pivot_col]
                self.tableau[i] -= coefficient * normalized_row

        self.tableau[pivot_row] = normalized_row
        return self.tableau

    def change_stage(self) -> np.ndarray:
        """Transition from the two-stage phase to the standard simplex phase.

        Removes the artificial variable columns and the auxiliary objective row,
        then resets stage-related attributes.

        Returns:
            np.ndarray: The updated tableau for the next stage.
        """
        if self.objectives:
            self.objectives.pop()

        if not self.objectives:
            return self.tableau

        # Remove artificial variable columns
        artificial_col_start = -self.n_artificial_vars - 1
        cols_to_remove = slice(artificial_col_start, -1)
        self.tableau = np.delete(self.tableau, cols_to_remove, axis=1)

        # Remove the auxiliary objective row
        self.tableau = np.delete(self.tableau, 0, axis=0)

        # Update tableau metadata
        self.n_cols = self.tableau.shape[1]
        self.n_stages = 1
        self.n_rows -= 1
        self.n_artificial_vars = 0
        self.stop_iter = False

        return self.tableau

    def run_simplex(self) -> Dict[str, Any]:
        """Execute the simplex algorithm on the tableau.

        Iterates through pivot operations until an optimal solution is found,
        the problem transitions between stages, or the iteration limit is reached.

        Returns:
            Dict[str, Any]: A dictionary mapping variable names to their optimal values,
                            including "P" for the objective value. Empty if limit exceeded.
        """
        for _ in range(self.MAX_ITERATIONS):
            if not self.objectives:
                return self._interpret_solution()

            pivot_row, pivot_col = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(pivot_row, pivot_col)

        return {}

    def _interpret_solution(self) -> Dict[str, Any]:
        """Extract the optimal solution from the final tableau.

        Identifies basic variables by finding columns with exactly one
        non-zero entry equal to 1, and reads off their values from the RHS.

        Returns:
            Dict[str, Any]: A dictionary with the objective value ("P") and the
                            values of each basic decision variable.
        """
        solution = {"P": abs(self.tableau[0, -1])}

        for i in range(self.n_vars):
            column = self.tableau[:, i]
            nonzero_rows = np.nonzero(column)[0]

            # A basic variable has exactly one non-zero entry equal to 1.0
            if len(nonzero_rows) == 1:
                row = nonzero_rows[0]
                if column[row] == 1.0:
                    solution[self.col_titles[i]] = self.tableau[row, -1]

        return solution


if __name__ == "__main__":
    import doctest

    doctest.testmod()
