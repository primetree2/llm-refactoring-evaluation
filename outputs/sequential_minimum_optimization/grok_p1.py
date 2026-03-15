"""
Support Vector Machine implementation using a simplified SMO algorithm.

This module implements a binary SVM classifier trained with a basic
Sequential Minimal Optimization (SMO) approach, along with kernel
support and visualization utilities.
"""

import os
import sys
import urllib.request
from contextlib import redirect_stdout
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


class SmoSVM:
    """
    Binary SVM classifier trained using a simplified SMO algorithm.
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
        self.train = train
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)
        self.tol = np.float64(tolerance if tolerance > 0.0001 else 0.001)
        self.auto_norm = auto_norm

        self.tags = train[:, 0].astype(np.float64)
        features = train[:, 1:].astype(np.float64)
        self.samples = self._norm(features) if self.auto_norm else features

        self.alphas = (
            np.zeros(len(self.samples), dtype=np.float64)
            if alpha_list is None
            else np.asarray(alpha_list, dtype=np.float64)
        )

        self.err = np.zeros(len(self.samples), dtype=np.float64)
        self._eps = 0.001
        self.indexes = list(range(len(self.samples)))
        self.kmat = self._build_kernel_matrix()

    def _build_kernel_matrix(self) -> np.ndarray:
        """Build the full kernel matrix for all training samples."""
        n = len(self.samples)
        kmat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                kmat[i, j] = self.kernel(self.samples[i], self.samples[j])
        return kmat

    def _k(self, i: int, j: Union[int, np.ndarray]) -> float:
        """Return kernel value between sample i and j (index or vector)."""
        if isinstance(j, np.ndarray):
            return float(self.kernel(self.samples[i], j))
        return float(self.kmat[i, j])

    def _is_unbound(self, i: int) -> bool:
        """Return whether alpha[i] is strictly between 0 and C."""
        return 0 < self.alphas[i] < self.c

    def _e(self, i: int) -> float:
        """Compute prediction error for sample i."""
        if self._is_unbound(i):
            return float(self.err[i])
        gx = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return float(gx - self.tags[i])

    def _violates_kkt(self, i: int) -> bool:
        """Check if sample i violates the KKT conditions."""
        r = self._e(i) * self.tags[i]
        if (r < -self.tol and self.alphas[i] < self.c) or (
            r > self.tol and self.alphas[i] > 0
        ):
            return True
        return False

    def fit(self) -> None:
        """Train the model using the simplified SMO loop."""
        changed = True
        while changed:
            changed = False
            for i1 in self.indexes:
                if not self._violates_kkt(i1):
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

                    if s == -1:
                        L = max(0.0, a2 - a1)
                        H = min(float(self.c), self.c + a2 - a1)
                    else:
                        L = max(0.0, a1 + a2 - self.c)
                        H = min(float(self.c), a1 + a2)

                    if L == H:
                        continue

                    k11 = self._k(i1, i1)
                    k22 = self._k(i2, i2)
                    k12 = self._k(i1, i2)
                    eta = k11 + k22 - 2 * k12

                    if eta <= 0:
                        continue

                    a2_new = a2 + (y2 * (e1 - e2)) / eta
                    a2_new = max(L, min(H, a2_new))
                    a1_new = a1 + s * (a2 - a2_new)

                    self.alphas[i1] = a1_new
                    self.alphas[i2] = a2_new

                    b1 = (
                        -e1
                        - y1 * k11 * (a1_new - a1)
                        - y2 * k12 * (a2_new - a2)
                        + self.b
                    )
                    b2 = (
                        -e2
                        - y1 * k12 * (a1_new - a1)
                        - y2 * k22 * (a2_new - a2)
                        + self.b
                    )

                    if 0 < a1_new < self.c:
                        self.b = b1
                    elif 0 < a2_new < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """Predict labels or decision values for test samples."""
        test_array = self._norm(test) if self.auto_norm else np.asarray(test)

        out = []
        n_samples = len(self.samples)
        for s in test_array:
            v = sum(
                self.alphas[i] * self.tags[i] * self._k(i, s)
                for i in range(n_samples)
            )
            v += self.b
            if classify:
                out.append(1 if v > 0 else -1)
            else:
                out.append(v)
        return np.array(out)

    def _norm(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization using training set statistics."""
        data = np.asarray(data, dtype=np.float64)
        if not hasattr(self, "min"):
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
        return (data - self.min) / (self.max - self.min)

    @property
    def support(self) -> List[int]:
        """Return indices of all support vectors (where alpha > 0)."""
        return [i for i, alpha in enumerate(self.alphas) if alpha > 0]


class Kernel:
    """Callable wrapper for common SVM kernel functions."""

    def __init__(
        self,
        kernel: str,
        degree: float = 1.0,
        coef0: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        self.name = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.name == "linear":
            return float(np.inner(v1, v2) + self.coef0)
        if self.name == "poly":
            return float((self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree)
        if self.name == "rbf":
            return float(np.exp(-1 * (self.gamma * np.linalg.norm(v1 - v2) ** 2)))
        return float(np.inner(v1, v2))


def _download_if_needed() -> None:
    """Download the breast cancer dataset if not already present."""
    if os.path.exists("cancer.csv"):
        return

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla"})
    with urllib.request.urlopen(req) as response:
        data = response.read().decode()

    with open("cancer.csv", "w", encoding="utf-8") as f:
        f.write(data)


def test_cancer() -> None:
    """Run SVM on the UCI breast cancer dataset and report accuracy."""
    _download_if_needed()

    df = pd.read_csv("cancer.csv", header=None, dtype={0: str})
    df = df.drop(columns=[df.columns[0]])
    df = df.dropna(axis=0)
    df = df.replace({"M": 1.0, "B": -1.0})

    arr = np.array(df, dtype=np.float64)
    train = arr[:328, :]
    test = arr[328:, :]

    test_tags = test[:, 0]
    test_x = test[:, 1:]

    kernel = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    model = SmoSVM(train, kernel, cost=0.4, auto_norm=True)
    model.fit()

    pred = model.predict(test_x)
    accuracy = np.mean(pred == test_tags)
    print("accuracy", accuracy)


def demo() -> None:
    """Generate demonstration plots for linear and RBF kernels."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

            _demo_linear(axs[0, 0], 0.1)
            _demo_linear(axs[0, 1], 500)
            _demo_rbf(axs[1, 0], 0.1)
            _demo_rbf(axs[1, 1], 500)

    print("plot ready")
    plt.show()


def _demo_linear(ax: plt.Axes, c: float) -> None:
    """Train linear kernel SVM on blobs and plot decision boundary."""
    x, y = make_blobs(
        n_samples=500, centers=2, n_features=2, random_state=1
    )
    y = np.where(y == 0, -1, 1).astype(np.float64)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    data = np.column_stack((y, x))

    kernel = Kernel("linear")
    model = SmoSVM(data, kernel, cost=c, auto_norm=False)
    model.fit()
    _plot(model, data, ax)


def _demo_rbf(ax: plt.Axes, c: float) -> None:
    """Train RBF kernel SVM on circles and plot decision boundary."""
    x, y = make_circles(
        n_samples=500, noise=0.1, factor=0.1, random_state=1
    )
    y = np.where(y == 0, -1, 1).astype(np.float64)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    data = np.column_stack((y, x))

    kernel = Kernel("rbf")
    model = SmoSVM(data, kernel, cost=c, auto_norm=False)
    model.fit()
    _plot(model, data, ax)


def _plot(model: SmoSVM, data: np.ndarray, ax: plt.Axes, res: int = 100) -> None:
    """Plot data points, decision boundary, and support vectors."""
    x = data[:, 1]
    y = data[:, 2]
    t = data[:, 0]

    xr = np.linspace(x.min(), x.max(), res)
    yr = np.linspace(y.min(), y.max(), res)
    pts = np.array([(a, b) for a in xr for b in yr], dtype=np.float64)

    pred = model.predict(pts, classify=False)
    grid = pred.reshape((res, res))

    ax.contour(xr, yr, grid.T, levels=(-1, 0, 1))
    ax.scatter(x, y, c=t, cmap=plt.cm.Dark2, alpha=0.5)

    sup_idx = model.support
    ax.scatter(x[sup_idx], y[sup_idx], c=t[sup_idx], cmap=plt.cm.Dark2, edgecolors="k")


if __name__ == "__main__":
    test_cancer()
    demo()
