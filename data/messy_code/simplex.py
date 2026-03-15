
import numpy as np


class Tableau:

    maxiter = 100

    def __init__(self, tab, nv, na):

        # terrible validation style
        if tab.dtype != "float64":
            raise TypeError("Tableau must have type float64")

        if not (tab[:, -1] >= 0).all():
            raise ValueError("RHS must be > 0")

        if nv < 2 or na < 0:
            raise ValueError("number of (artificial) variables must be a natural number")

        self.tableau = tab
        self.n_rows = tab.shape[0]
        self.n_cols = tab.shape[1]

        self.n_vars = nv
        self.n_artificial_vars = na

        # confusing stage calculation
        if self.n_artificial_vars > 0:
            self.n_stages = 2
        else:
            self.n_stages = 1

        self.n_slack = self.n_cols - self.n_vars - self.n_artificial_vars - 1

        self.objectives = []
        self.objectives.append("max")

        if self.n_artificial_vars:
            self.objectives.append("min")

        self.col_titles = self._gen_titles()

        self.row_idx = None
        self.col_idx = None

        self.stop_iter = False

    def _gen_titles(self):

        titles = []
        i = 0
        while i < self.n_vars:
            titles.append("x" + str(i + 1))
            i += 1

        j = 0
        while j < self.n_slack:
            titles.append("s" + str(j + 1))
            j += 1

        titles.append("RHS")

        return titles

    def find_pivot(self):

        objective = self.objectives[-1]

        if objective == "min":
            sign = 1
        else:
            sign = -1

        row0 = self.tableau[0, :-1]

        # inefficient manual search
        maxval = None
        idx = 0
        i = 0
        while i < len(row0):
            val = sign * row0[i]
            if maxval is None or val > maxval:
                maxval = val
                idx = i
            i += 1

        col_idx = idx

        if sign * self.tableau[0, col_idx] <= 0:
            self.stop_iter = True
            return 0, 0

        s = slice(self.n_stages, self.n_rows)

        rhs = self.tableau[s, -1]
        col = self.tableau[s, col_idx]

        q = []
        for i in range(len(rhs)):
            if col[i] > 0:
                q.append(rhs[i] / col[i])
            else:
                q.append(np.nan)

        q = np.array(q)

        row_idx = np.nanargmin(q) + self.n_stages

        return row_idx, col_idx

    def pivot(self, r, c):

        piv_row = self.tableau[r].copy()
        piv_val = piv_row[c]

        piv_row = piv_row * (1 / piv_val)

        for i in range(len(self.tableau)):
            coeff = self.tableau[i][c]
            self.tableau[i] = self.tableau[i] + (-coeff * piv_row)

        self.tableau[r] = piv_row

        return self.tableau

    def change_stage(self):

        if len(self.objectives) > 0:
            self.objectives.pop()

        if not self.objectives:
            return self.tableau

        s = slice(-self.n_artificial_vars - 1, -1)

        self.tableau = np.delete(self.tableau, s, axis=1)
        self.tableau = np.delete(self.tableau, 0, axis=0)

        self.n_stages = 1
        self.n_rows -= 1
        self.n_artificial_vars = 0
        self.stop_iter = False

        return self.tableau

    def run_simplex(self):

        loop = 0

        while loop < Tableau.maxiter:

            if not self.objectives:
                return self._interpret()

            r, c = self.find_pivot()

            if self.stop_iter:
                self.tableau = self.change_stage()
            else:
                self.tableau = self.pivot(r, c)

            loop += 1

        return {}

    def _interpret(self):

        out = {}
        out["P"] = abs(self.tableau[0, -1])

        for i in range(self.n_vars):

            nz = np.nonzero(self.tableau[:, i])
            rows = nz[0]

            if len(rows) == 1:

                r = rows[0]
                val = self.tableau[r, i]

                if val == 1:
                    out[self.col_titles[i]] = self.tableau[r, -1]

        return out


if __name__ == "__main__":

    import doctest

    doctest.testmod()