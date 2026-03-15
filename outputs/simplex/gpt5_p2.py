import numpy as np
from typing import Dict, List, Optional, Tuple


class Tableau:
    """Simplex tableau for solving linear programs (supports two-phase simplex).

    The tableau is expected to be in standard simplex tableau form where:
    - Row 0 is the objective function row for the current stage/phase.
    - The last column is the RHS (all RHS values must be non-negative).
    - If artificial variables are present, the algorithm runs in two stages.

    Notes:
        This implementation preserves the original behavior and public API.
    """

    # Kept for backward compatibility (original attribute name).
    maxiter: int = 100

    def __init__(self, tab: np.ndarray, nv: int, na: int) -> None:
        """Create a tableau instance.

        Args:
            tab: 2D NumPy array representing the simplex tableau. Must be float64.
            nv: Number of decision variables.
            na: Number of artificial variables.

        Raises:
            TypeError: If `tab` is not float64.
            ValueError: If RHS contains negative values or counts are invalid.
        """
        # Keep original validation behavior/messages.
        if tab.dtype != "float64":
            raise TypeError("Tableau must have type float64")

        if not (tab[:, -1] >= 0).all():
            raise ValueError("RHS must be > 0")

        if nv < 2 or na < 0:
            raise ValueError("number of (artificial) variables must be a natural number")

        self.tableau: np.ndarray = tab
        self.n_rows: int = tab.shape[0]
        self.n_cols: int = tab.shape[1]

        self.n_vars: int = nv
        self.n_artificial_vars: int = na

        # Two-phase simplex uses two stages when artificial variables exist.
        self.n_stages: int = 2 if self.n_artificial_vars > 0 else 1

        # Slack variable count is inferred from tableau layout.
        self.n_slack: int = self.n_cols - self.n_vars - self.n_artificial_vars - 1

        # Objective stack: last item indicates the current stage objective.
        # Original behavior: always start with "max", then append "min" if Phase I needed.
        self.objectives: List[str] = ["max"]
        if self.n_artificial_vars:
            self.objectives.append("min")

        self.col_titles: List[str] = self._gen_titles()

        # Present in original code; not used by algorithm directly.
        self.row_idx: Optional[int] = None
        self.col_idx: Optional[int] = None

        # Flag used to signal when the current stage is optimal (no improving pivot).
        self.stop_iter: bool = False

    def _gen_titles(self) -> List[str]:
        """Generate column titles for decision variables, slack variables, and RHS.

        Returns:
            List of column labels (e.g., ["x1", "x2", ..., "s1", ..., "RHS"]).
        """
        titles: List[str] = []

        for i in range(self.n_vars):
            titles.append(f"x{i + 1}")

        for j in range(self.n_slack):
            titles.append(f"s{j + 1}")

        titles.append("RHS")
        return titles

    def find_pivot(self) -> Tuple[int, int]:
        """Find the next pivot position.

        The entering variable is chosen from the objective row:
        - If current objective is "min": choose the largest positive coefficient.
        - If current objective is "max": choose the most negative coefficient.
        Then, the leaving variable is chosen via the minimum ratio test.

        Returns:
            (pivot_row_index, pivot_col_index). If no improving pivot exists,
            sets `self.stop_iter = True` and returns (0, 0).
        """
        objective = self.objectives[-1]
        sign = 1 if objective == "min" else -1

        # Choose pivot column: maximize sign * coefficient.
        obj_coeffs = self.tableau[0, :-1]
        col_idx = int(np.argmax(sign * obj_coeffs))

        # Optimality check: if the best candidate doesn't improve the objective.
        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        # Consider only constraint rows (skip objective rows for multi-stage).
        constraint_rows = slice(self.n_stages, self.n_rows)
        rhs = self.tableau[constraint_rows, -1]
        col = self.tableau[constraint_rows, col_idx]

        # Minimum ratio test: rhs / col for col > 0, otherwise NaN.
        ratios = np.where(col > 0, rhs / col, np.nan)
        row_idx = int(np.nanargmin(ratios)) + self.n_stages

        return row_idx, col_idx

    def pivot(self, r: int, c: int) -> np.ndarray:
        """Perform a pivot operation around tableau[r, c].

        This is the standard Gauss-Jordan elimination step used by simplex.

        Args:
            r: Pivot row index.
            c: Pivot column index.

        Returns:
            The updated tableau.
        """
        pivot_row = self.tableau[r].copy()
        pivot_val = pivot_row[c]

        # Normalize pivot row so pivot element becomes 1.
        pivot_row *= 1 / pivot_val

        # Eliminate pivot column in all rows (including the pivot row; overwritten later).
        for i in range(self.n_rows):
            coeff = self.tableau[i, c]
            self.tableau[i] = self.tableau[i] + (-coeff * pivot_row)

        # Restore the normalized pivot row (numerical stability / correctness).
        self.tableau[r] = pivot_row
        return self.tableau

    def change_stage(self) -> np.ndarray:
        """Switch to the next stage (Phase II) by removing artificial variables.

        Behavior preserved from original implementation:
        - Pops the current objective from `self.objectives`.
        - If objectives remain, deletes artificial variable columns and removes
          the first row (previous objective row).
        - Resets stage-related counters and flags.

        Returns:
            The updated tableau.
        """
        if len(self.objectives) > 0:
            self.objectives.pop()

        if not self.objectives:
            return self.tableau

        # Remove artificial variable columns just before RHS.
        artificial_cols = slice(-self.n_artificial_vars - 1, -1)
        self.tableau = np.delete(self.tableau, artificial_cols, axis=1)

        # Remove the objective row for the completed stage.
        self.tableau = np.delete(self.tableau, 0, axis=0)

        # Update bookkeeping (mirrors original behavior).
        self.n_stages = 1
        self.n_rows -= 1
        self.n_artificial_vars = 0
        self.stop_iter = False

        return self.tableau

    def run_simplex(self) -> Dict[str, float]:
        """Run simplex iterations until all stages are complete or max iterations hit.

        Returns:
            A dictionary containing the objective value under "P" and basic
            decision variables ("x1", "x2", ...). Returns {} on iteration limit.
        """
        loop = 0

        while loop < Tableau.maxiter:
            # If all objectives have been processed, interpret final tableau.
            if not self.objectives:
                return self._interpret()

            r, c = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(r, c)

            loop += 1

        # Iteration limit hit; preserve original empty result behavior.
        return {}

    def _interpret(self) -> Dict[str, float]:
        """Interpret the tableau to extract the objective value and basic variables.

        Returns:
            Mapping of solution values. "P" holds the objective value.
        """
        out: Dict[str, float] = {"P": abs(self.tableau[0, -1])}

        # A decision variable is basic if its column has a single non-zero,
        # and that non-zero entry is exactly 1 (preserves original semantics).
        for i in range(self.n_vars):
            rows = np.nonzero(self.tableau[:, i])[0]
            if len(rows) == 1:
                r = int(rows[0])
                if self.tableau[r, i] == 1:
                    out[self.col_titles[i]] = float(self.tableau[r, -1])

        return out


if __name__ == "__main__":
    import doctest

    doctest.testmod()
