import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


ArrayLike = Union[np.ndarray, Sequence[float]]


class SmoSVM:
    """
    Minimal SMO-based Support Vector Machine (binary classification).

    Expected training data format:
        - train[:, 0] -> labels in {-1, +1}
        - train[:, 1:] -> features

    Notes:
        - Implements a simplified SMO optimization loop.
        - Error cache is present but not actively updated (kept to preserve behavior).
        - Optional min-max normalization based on the training set.
    """

    def __init__(
        self,
        train: np.ndarray,
        kernel_func: Callable[[np.ndarray, np.ndarray], float],
        alpha_list: Optional[ArrayLike] = None,
        cost: float = 0.4,
        b: float = 0.0,
        tolerance: float = 0.001,
        auto_norm: bool = True,
    ) -> None:
        self.train = train
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)

        # Preserve original tolerance guard.
        self.tol = np.float64(tolerance if tolerance > 0.0001 else 0.001)

        self.auto_norm = auto_norm

        self.tags = self.train[:, 0].astype(np.float64)

        features = self.train[:, 1:].astype(np.float64)
        self.samples = self._norm(features) if self.auto_norm else features

        if alpha_list is None:
            self.alphas = np.zeros(self.train.shape[0], dtype=np.float64)
        else:
            self.alphas = np.asarray(alpha_list, dtype=np.float64)

        # Present for compatibility; cache is not updated in the original algorithm.
        self.err = np.zeros(len(self.samples), dtype=np.float64)
        self._eps = 0.001  # Kept for compatibility; unused in current implementation.

        self.indexes = list(range(len(self.samples)))
        self.kmat = self._build_kernel_matrix()

    def _build_kernel_matrix(self) -> np.ndarray:
        """Precompute the kernel matrix K[i, j] for all training samples."""
        n = len(self.samples)
        kmat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                kmat[i, j] = self.kernel(self.samples[i], self.samples[j])
        return kmat

    def _k(self, i: int, j: Union[int, np.ndarray]) -> float:
        """
        Kernel lookup.

        - If `j` is an ndarray: compute kernel(sample_i, j)
        - Else: return precomputed kernel matrix entry K[i, j]
        """
        if isinstance(j, np.ndarray):
            return float(self.kernel(self.samples[i], j))
        return float(self.kmat[i, j])

    def _is_unbound(self, i: int) -> bool:
        """Return True if alpha_i is strictly between 0 and C."""
        return 0.0 < self.alphas[i] < self.c

    def _e(self, i: int) -> float:
        """
        Compute error E_i = f(x_i) - y_i.

        Uses the (unused) error cache when alpha is unbound, matching original behavior.
        """
        if self._is_unbound(i):
            return float(self.err[i])

        gx = float(np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b)
        return gx - float(self.tags[i])

    def _violates_kkt(self, i: int) -> bool:
        """
        Return True if sample i violates KKT conditions (as defined in the original code).
        """
        r = self._e(i) * float(self.tags[i])
        return (r < -self.tol and self.alphas[i] < self.c) or (r > self.tol and self.alphas[i] > 0.0)

    def fit(self) -> None:
        """Train the SVM using a simplified SMO optimization loop."""
        changed = True

        while changed:
            changed = False

            for i1 in self.indexes:
                if not self._violates_kkt(i1):
                    continue

                for i2 in self.indexes:
                    if i1 == i2:
                        continue

                    y1 = float(self.tags[i1])
                    y2 = float(self.tags[i2])

                    a1 = float(self.alphas[i1])
                    a2 = float(self.alphas[i2])

                    e1 = float(self._e(i1))
                    e2 = float(self._e(i2))

                    s = y1 * y2

                    # Compute bounds for a2 update.
                    if s == -1.0:
                        L = max(0.0, a2 - a1)
                        H = min(float(self.c), float(self.c) + a2 - a1)
                    else:
                        L = max(0.0, a1 + a2 - float(self.c))
                        H = min(float(self.c), a1 + a2)

                    if L == H:
                        continue

                    k11 = self._k(i1, i1)
                    k22 = self._k(i2, i2)
                    k12 = self._k(i1, i2)

                    eta = k11 + k22 - 2.0 * k12
                    if eta <= 0.0:
                        continue

                    a2_new = a2 + (y2 * (e1 - e2)) / eta
                    if a2_new > H:
                        a2_new = H
                    if a2_new < L:
                        a2_new = L

                    a1_new = a1 + s * (a2 - a2_new)

                    self.alphas[i1] = a1_new
                    self.alphas[i2] = a2_new

                    b1 = -e1 - y1 * k11 * (a1_new - a1) - y2 * k12 * (a2_new - a2) + float(self.b)
                    b2 = -e2 - y1 * k12 * (a1_new - a1) - y2 * k22 * (a2_new - a2) + float(self.b)

                    if 0.0 < a1_new < float(self.c):
                        self.b = np.float64(b1)
                    elif 0.0 < a2_new < float(self.c):
                        self.b = np.float64(b2)
                    else:
                        self.b = np.float64((b1 + b2) / 2.0)

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """
        Predict for given samples.

        Args:
            test: Feature matrix of shape (n_samples, n_features).
            classify: If True, return labels {-1, +1}. If False, return decision values.

        Returns:
            A numpy array of predictions (labels or decision values).
        """
        test_x = self._norm(test) if self.auto_norm else test

        out: List[float] = []
        for sample in test_x:
            v = 0.0
            for i in range(len(self.samples)):
                v += float(self.alphas[i]) * float(self.tags[i]) * self._k(i, sample)
            v += float(self.b)

            if classify:
                out.append(1.0 if v > 0.0 else -1.0)
            else:
                out.append(v)

        return np.asarray(out, dtype=np.float64)

    def _norm(self, data: np.ndarray) -> np.ndarray:
        """
        Min-max normalize features based on training set statistics.

        Training set stats are initialized on first call.
        """
        if not hasattr(self, "min"):
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
        return (data - self.min) / (self.max - self.min)

    @property
    def support(self) -> List[int]:
        """Return indices of support vectors where alpha_i > 0."""
        return [i for i, a in enumerate(self.alphas) if a > 0.0]


class Kernel:
    """Callable kernel function wrapper for SVM."""

    def __init__(self, kernel: str, degree: float = 1.0, coef0: float = 0.0, gamma: float = 1.0) -> None:
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
            return float(np.exp(-1.0 * (self.gamma * (np.linalg.norm(v1 - v2) ** 2))))

        # Fallback to linear.
        return float(np.inner(v1, v2))


def _download_text(url: str, out_path: Path) -> None:
    """Download URL content and write it to `out_path`."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla"})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode()

    out_path.write_text(text, encoding="utf-8")


def test_cancer() -> None:
    """
    Train and evaluate the SMO SVM on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.
    Downloads the dataset locally if not present as `cancer.csv`.
    """
    data_path = Path("cancer.csv")
    if not data_path.exists():
        _download_text(URL, data_path)

    df = pd.read_csv(data_path, header=None, dtype={0: str})

    # Drop ID column, keep diagnosis label + features.
    df = df.drop(columns=[0]).dropna(axis=0)

    # Map labels to +/-1.
    df = df.replace({"M": 1.0, "B": -1.0})

    arr = np.asarray(df, dtype=np.float64)

    train = arr[:328, :]
    test = arr[328:, :]

    test_tags = test[:, 0]
    test_x = test[:, 1:]

    ker = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    init_alphas = np.zeros(train.shape[0], dtype=np.float64)

    model = SmoSVM(train, ker, init_alphas)
    model.fit()

    pred = model.predict(test_x)

    good = 0
    for i in range(len(pred)):
        if pred[i] == test_tags[i]:
            good += 1

    print("accuracy", good / len(pred))


def demo() -> None:
    """Generate demo plots for linear and RBF kernels under two different C values."""
    sys.stdout = open(os.devnull, "w")

    fig, axs = plt.subplots(2, 2)

    _demo_linear(axs[0][0], 0.1)
    _demo_linear(axs[0][1], 500)
    _demo_rbf(axs[1][0], 0.1)
    _demo_rbf(axs[1][1], 500)

    sys.stdout = sys.__stdout__
    print("plot ready")


def _demo_linear(ax, c: float) -> None:
    """Demo SVM on linearly separable blobs."""
    x, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=1)
    y = y.astype(np.float64)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)

    data = np.hstack((y.reshape(500, 1), x))

    kernel = Kernel("linear")
    model = SmoSVM(data, kernel, cost=c, auto_norm=False)
    model.fit()

    _plot(model, data, ax)


def _demo_rbf(ax, c: float) -> None:
    """Demo SVM on non-linearly separable circles."""
    x, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
    y = y.astype(np.float64)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)

    data = np.hstack((y.reshape(500, 1), x))

    kernel = Kernel("rbf")
    model = SmoSVM(data, kernel, cost=c, auto_norm=False)
    model.fit()

    _plot(model, data, ax)


def _plot(model: SmoSVM, data: np.ndarray, ax, res: int = 100) -> None:
    """Plot decision contours and support vectors."""
    x = data[:, 1]
    y = data[:, 2]
    t = data[:, 0]

    xr = np.linspace(x.min(), x.max(), res)
    yr = np.linspace(y.min(), y.max(), res)

    pts = np.array([(a, b) for a in xr for b in yr], dtype=np.float64).reshape(res * res, 2)
    pred = model.predict(pts, classify=False)

    grid = pred.reshape((len(xr), len(yr)))

    ax.contour(xr, yr, grid.T, levels=(-1, 0, 1))
    ax.scatter(x, y, c=t, cmap=plt.cm.Dark2, alpha=0.5)

    sup = model.support
    ax.scatter(x[sup], y[sup], c=t[sup], cmap=plt.cm.Dark2)


if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
