"""
Simple SVM implementation using a simplified SMO optimizer.

This module includes:
- A Kernel callable for linear, polynomial, and RBF kernels
- An SVM trainer/predictor (SmoSVM)
- A dataset test using the UCI WDBC breast cancer dataset
- Demo plots for linear and RBF kernels on synthetic data
"""

from __future__ import annotations

import os
import sys
import urllib.request
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
CANCER_CSV_PATH = "cancer.csv"


class SmoSVM:
    """SVM classifier trained with a simplified SMO-style loop.

    Notes:
        - Expects training data as a numpy array where the first column is the label (±1)
          and remaining columns are features.
        - Uses a precomputed kernel matrix for training samples.
        - Error caching is implemented but not actively updated in the provided algorithm
          (kept to preserve original behavior).
        - Optionally applies min-max normalization using statistics derived from training data.
    """

    def __init__(
        self,
        train: np.ndarray,
        kernel_func: Callable[[np.ndarray, np.ndarray], float],
        alpha_list: Optional[np.ndarray] = None,
        cost: float = 0.4,
        b: float = 0.0,
        tolerance: float = 0.001,
        auto_norm: bool = True,
    ) -> None:
        """Initialize the SVM model.

        Args:
            train: Training data [n_samples, 1 + n_features], first column is label.
            kernel_func: Kernel function (callable).
            alpha_list: Optional initial alpha values (length n_samples).
            cost: Regularization parameter C.
            b: Initial bias term.
            tolerance: KKT tolerance.
            auto_norm: If True, apply min-max normalization to features.
        """
        self.train = train
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)

        # Keep original tolerance correction behavior.
        self.tol = np.float64(tolerance)
        if tolerance <= 0.0001:
            self.tol = np.float64(0.001)

        self.auto_norm = auto_norm

        # Labels (y) and feature matrix (X)
        self.tags = train[:, 0]
        self.samples = self._norm(train[:, 1:]) if self.auto_norm else train[:, 1:]

        # Lagrange multipliers (alphas)
        self.alphas = np.zeros(train.shape[0]) if alpha_list is None else alpha_list

        # Error cache (kept for parity with original implementation)
        self.err = np.zeros(len(self.samples))
        self._eps = 0.001  # Present in original code; not used elsewhere.

        self.indexes = list(range(len(self.samples)))

        # Precompute training kernel matrix K[i, j] = k(x_i, x_j)
        self.kmat = self._build_k()

        # Present in original code; unused elsewhere (kept for compatibility).
        self.unbound: List[int] = []

    def _build_k(self) -> np.ndarray:
        """Compute the full kernel matrix for training samples."""
        n = len(self.samples)
        kmat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                kmat[i][j] = self.kernel(self.samples[i], self.samples[j])

        return kmat

    def _k(self, i: int, j: Union[int, np.ndarray]) -> float:
        """Kernel access helper.

        Args:
            i: Index of a training sample.
            j: Either another training index or a raw feature vector.

        Returns:
            Kernel value.
        """
        if isinstance(j, np.ndarray):
            return self.kernel(self.samples[i], j)
        return self.kmat[i][j]

    def _is_unbound(self, i: int) -> bool:
        """Whether alpha_i is strictly within (0, C)."""
        return 0 < self.alphas[i] < self.c

    def _e(self, i: int) -> float:
        """Compute the error E_i = f(x_i) - y_i."""
        # Original behavior: return cached error only if unbound.
        if self._is_unbound(i):
            return self.err[i]

        gx = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return gx - self.tags[i]

    def _check_kkt(self, i: int) -> bool:
        """Return True if the KKT conditions are violated (needs optimization)."""
        r = self._e(i) * self.tags[i]
        return (r < -self.tol and self.alphas[i] < self.c) or (r > self.tol and self.alphas[i] > 0)

    def fit(self) -> None:
        """Train the SVM using a simplified SMO approach."""
        changed = True

        # Continue until a full pass makes no changes.
        while changed:
            changed = False

            for i1 in self.indexes:
                if not self._check_kkt(i1):
                    continue

                for i2 in self.indexes:
                    if i1 == i2:
                        continue

                    y1 = self.tags[i1]
                    y2 = self.tags[i2]

                    a1 = self.alphas[i1]
                    a2 = self.alphas[i2]

                    e1 = self._e(i1)
                    e2 = self._e(i2)

                    s = y1 * y2

                    # Compute box constraints L and H for alpha2 update.
                    if s == -1:
                        L = max(0, a2 - a1)
                        H = min(self.c, self.c + a2 - a1)
                    else:
                        L = max(0, a1 + a2 - self.c)
                        H = min(self.c, a1 + a2)

                    if L == H:
                        continue

                    k11 = self._k(i1, i1)
                    k22 = self._k(i2, i2)
                    k12 = self._k(i1, i2)

                    eta = k11 + k22 - 2 * k12
                    if eta <= 0:
                        continue

                    # Update alpha2, then alpha1.
                    a2new = a2 + (y2 * (e1 - e2)) / eta

                    # Clip alpha2 into [L, H].
                    if a2new > H:
                        a2new = H
                    if a2new < L:
                        a2new = L

                    a1new = a1 + s * (a2 - a2new)

                    # Store updated alphas.
                    self.alphas[i1] = a1new
                    self.alphas[i2] = a2new

                    # Update bias term b using standard SMO formulas.
                    b1 = -e1 - y1 * k11 * (a1new - a1) - y2 * k12 * (a2new - a2) + self.b
                    b2 = -e2 - y1 * k12 * (a1new - a1) - y2 * k22 * (a2new - a2) + self.b

                    if 0 < a1new < self.c:
                        self.b = b1
                    elif 0 < a2new < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """Predict labels or decision function values for given samples.

        Args:
            test: Feature matrix [n_samples, n_features].
            classify: If True, return class labels (±1). If False, return raw scores.

        Returns:
            Predictions as a numpy array.
        """
        if self.auto_norm:
            test = self._norm(test)

        out: List[float] = []

        for s in test:
            v = 0.0
            for i in range(len(self.samples)):
                v += self.alphas[i] * self.tags[i] * self._k(i, s)
            v += self.b

            if classify:
                out.append(1 if v > 0 else -1)
            else:
                out.append(v)

        return np.array(out)

    def _norm(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalize features based on training-set min/max.

        Note:
            Keeps original behavior, including potential divide-by-zero when max == min.
        """
        if not hasattr(self, "min"):
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)

        return (data - self.min) / (self.max - self.min)

    @property
    def support(self) -> List[int]:
        """Indices of support vectors (alpha > 0)."""
        return [i for i in range(len(self.alphas)) if self.alphas[i] > 0]


class Kernel:
    """Kernel function wrapper supporting 'linear', 'poly', and 'rbf'."""

    def __init__(self, kernel: str, degree: float = 1.0, coef0: float = 0.0, gamma: float = 1.0) -> None:
        """
        Args:
            kernel: Name of kernel ('linear', 'poly', 'rbf').
            degree: Polynomial degree (for 'poly').
            coef0: Bias term (for 'linear' and 'poly').
            gamma: Gamma parameter (for 'rbf' and 'poly').
        """
        self.name = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute the kernel value between two vectors."""
        if self.name == "linear":
            return np.inner(v1, v2) + self.coef0

        if self.name == "poly":
            return (self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree

        if self.name == "rbf":
            return np.exp(-1 * (self.gamma * np.linalg.norm(v1 - v2) ** 2))

        # Default to linear inner product for unknown names (original behavior).
        return np.inner(v1, v2)


def _download_cancer_csv(path: str = CANCER_CSV_PATH) -> None:
    """Download the cancer dataset if not already present."""
    if os.path.exists(path):
        return

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla"})
    with urllib.request.urlopen(req) as response:
        data = response.read().decode()

    with open(path, "w", encoding="utf-8") as f:
        f.write(data)


def test_cancer() -> None:
    """Train/test the SMO SVM on the UCI breast cancer dataset and print accuracy."""
    _download_cancer_csv(CANCER_CSV_PATH)

    df = pd.read_csv(CANCER_CSV_PATH, header=None, dtype={0: str})

    # Drop ID column (first column in the file).
    del df[df.columns[0]]

    df = df.dropna(axis=0)
    df = df.replace({"M": 1.0, "B": -1.0})

    arr = np.array(df)

    train = arr[:328, :]
    test = arr[328:, :]

    test_tags = test[:, 0]
    test_x = test[:, 1:]

    ker = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    a = np.zeros(train.shape[0])

    model = SmoSVM(train, ker, a)
    model.fit()

    pred = model.predict(test_x)

    good = 0
    for i in range(len(pred)):
        if pred[i] == test_tags[i]:
            good += 1

    print("accuracy", good / len(pred))


def demo() -> None:
    """Generate demo plots for linear and RBF SVM on synthetic datasets."""
    # Suppress console output during demos (preserve original behavior).
    devnull = open(os.devnull, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = devnull

    try:
        _, axs = plt.subplots(2, 2)
        _demo_linear(axs[0][0], 0.1)
        _demo_linear(axs[0][1], 500)
        _demo_rbf(axs[1][0], 0.1)
        _demo_rbf(axs[1][1], 500)
    finally:
        sys.stdout = original_stdout
        devnull.close()

    print("plot ready")


def _demo_linear(ax: plt.Axes, c: float) -> None:
    """Plot SVM decision boundaries using a linear kernel on blob data."""
    x, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=1)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)

    data = np.hstack((y.reshape(500, 1), x))

    k = Kernel("linear")
    m = SmoSVM(data, k, cost=c, auto_norm=False)
    m.fit()

    _plot(m, data, ax)


def _demo_rbf(ax: plt.Axes, c: float) -> None:
    """Plot SVM decision boundaries using an RBF kernel on circles data."""
    x, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)

    data = np.hstack((y.reshape(500, 1), x))

    k = Kernel("rbf")
    m = SmoSVM(data, k, cost=c, auto_norm=False)
    m.fit()

    _plot(m, data, ax)


def _plot(model: SmoSVM, data: np.ndarray, ax: plt.Axes, res: int = 100) -> None:
    """Plot data points, support vectors, and the decision contours.

    Args:
        model: Trained SmoSVM.
        data: Dataset with labels in first column.
        ax: Matplotlib axes to draw onto.
        res: Grid resolution for decision function sampling.
    """
    x = data[:, 1]
    y = data[:, 2]
    t = data[:, 0]

    xr = np.linspace(x.min(), x.max(), res)
    yr = np.linspace(y.min(), y.max(), res)

    # Create grid points (res^2 x 2) in a deterministic order.
    pts = np.array([(a, b) for a in xr for b in yr]).reshape(res * res, 2)

    pred = model.predict(pts, False)
    grid = pred.reshape((len(xr), len(yr)))

    ax.contour(xr, yr, np.asmatrix(grid).T, levels=(-1, 0, 1))
    ax.scatter(x, y, c=t, cmap=plt.cm.Dark2, alpha=0.5)

    sup = model.support
    ax.scatter(x[sup], y[sup], c=t[sup], cmap=plt.cm.Dark2)


if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
