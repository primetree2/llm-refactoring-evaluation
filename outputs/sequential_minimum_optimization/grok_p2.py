import os
import sys
import urllib.request
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
DATA_FILE = "cancer.csv"


class Kernel:
    """Kernel function wrapper for SVM."""

    def __init__(
        self,
        kernel: str = "linear",
        degree: float = 1.0,
        coef0: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        """Initialize the kernel.

        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf')
            degree: Degree for polynomial kernel
            coef0: Independent term in kernel function
            gamma: Kernel coefficient for rbf and poly
        """
        self.name = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute kernel value between two vectors."""
        if self.name == "linear":
            return float(np.inner(v1, v2) + self.coef0)

        if self.name == "poly":
            return float((self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree)

        if self.name == "rbf":
            return float(np.exp(-self.gamma * np.linalg.norm(v1 - v2) ** 2))

        return float(np.inner(v1, v2))


class SmoSVM:
    """Support Vector Machine using a simplified Sequential Minimal Optimization (SMO) algorithm."""

    def __init__(
        self,
        train: np.ndarray,
        kernel_func: Kernel,
        alpha_list: Optional[np.ndarray] = None,
        cost: float = 0.4,
        b: float = 0.0,
        tolerance: float = 0.001,
        auto_norm: bool = True,
    ) -> None:
        """Initialize SMO SVM model.

        Args:
            train: Training data where first column contains labels (±1)
            kernel_func: Kernel instance to use
            alpha_list: Initial Lagrange multipliers (optional)
            cost: Regularization parameter C
            b: Initial bias term
            tolerance: Tolerance for KKT condition checking
            auto_norm: Whether to apply min-max scaling to features
        """
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)
        self.tol = np.float64(tolerance)
        self.auto_norm = auto_norm
        self._eps = 0.001

        if tolerance <= 0.0001:
            self.tol = np.float64(0.001)

        self.tags = train[:, 0].astype(np.float64)

        if self.auto_norm:
            self.samples = self._normalize(train[:, 1:])
        else:
            self.samples = train[:, 1:].astype(np.float64)

        if alpha_list is None:
            self.alphas = np.zeros(len(self.samples), dtype=np.float64)
        else:
            self.alphas = np.asarray(alpha_list, dtype=np.float64).copy()

        self.err = np.zeros(len(self.samples), dtype=np.float64)
        self.indexes = list(range(len(self.samples)))

        self.kmat = self._build_kernel_matrix()

    def _build_kernel_matrix(self) -> np.ndarray:
        """Build full kernel matrix K for all training samples."""
        n = len(self.samples)
        kmat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                kmat[i, j] = self.kernel(self.samples[i], self.samples[j])

        return kmat

    def _kernel(self, i: int, j: Union[int, np.ndarray]) -> float:
        """Return kernel value between sample i and j (or new vector)."""
        if isinstance(j, np.ndarray):
            return self.kernel(self.samples[i], j)
        return self.kmat[i, j]

    def _is_unbound(self, i: int) -> bool:
        """Check if alpha_i is strictly between 0 and C."""
        return 0 < self.alphas[i] < self.c

    def _compute_error(self, i: int) -> float:
        """Compute prediction error for sample i.

        Uses cached error for unbound variables when available.
        """
        if self._is_unbound(i):
            return self.err[i]

        gx = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return gx - self.tags[i]

    def _violates_kkt(self, i: int) -> bool:
        """Check if sample i violates the KKT conditions within tolerance."""
        r = self._compute_error(i) * self.tags[i]
        if (r < -self.tol and self.alphas[i] < self.c) or (
            r > self.tol and self.alphas[i] > 0
        ):
            return True
        return False

    def fit(self) -> None:
        """Train the SVM using simplified SMO algorithm."""
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
                    e1 = self._compute_error(i1)
                    e2 = self._compute_error(i2)
                    s = y1 * y2

                    if s == -1:
                        L = max(0.0, a2 - a1)
                        H = min(self.c, self.c + a2 - a1)
                    else:
                        L = max(0.0, a1 + a2 - self.c)
                        H = min(self.c, a1 + a2)

                    if L == H:
                        continue

                    k11 = self._kernel(i1, i1)
                    k22 = self._kernel(i2, i2)
                    k12 = self._kernel(i1, i2)

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
        """Predict on new samples.

        Args:
            test: Test samples (without labels)
            classify: If True, return class labels (±1), else return decision values

        Returns:
            Array of predictions
        """
        if self.auto_norm:
            test = self._normalize(test)

        predictions = []
        for sample in test:
            val = 0.0
            for i in range(len(self.samples)):
                val += self.alphas[i] * self.tags[i] * self._kernel(i, sample)
            val += self.b

            if classify:
                predictions.append(1 if val > 0 else -1)
            else:
                predictions.append(val)

        return np.array(predictions)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to [0, 1] range.

        Stores min and max from first call (training data).
        """
        data = np.asarray(data, dtype=np.float64)

        if not hasattr(self, "_min"):
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0)

        return (data - self._min) / (self._max - self._min + 1e-8)

    @property
    def support(self) -> List[int]:
        """Return indices of support vectors (where alpha > 0)."""
        return [i for i, alpha in enumerate(self.alphas) if alpha > 0]


def test_cancer() -> None:
    """Test the SVM on the Breast Cancer Wisconsin dataset."""
    if not os.path.exists(DATA_FILE):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla"})
        with urllib.request.urlopen(req) as response:
            data = response.read().decode()

        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(data)

    df = pd.read_csv(DATA_FILE, header=None, dtype={0: str})
    del df[df.columns[0]]  # Remove ID column
    df = df.dropna(axis=0)
    df = df.replace({"M": 1.0, "B": -1.0})

    arr = np.array(df, dtype=np.float64)
    train = arr[:328, :]
    test = arr[328:, :]

    test_labels = test[:, 0]
    test_features = test[:, 1:]

    kernel = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    model = SmoSVM(train, kernel, cost=0.4, auto_norm=True)

    model.fit()
    predictions = model.predict(test_features)

    accuracy = np.mean(predictions == test_labels)
    print(f"accuracy {accuracy:.4f}")


def demo() -> None:
    """Run demo visualizations with linear and RBF kernels."""
    # Suppress output during plotting
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    try:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        _demo_linear(axs[0, 0], 0.1)
        _demo_linear(axs[0, 1], 500)
        _demo_rbf(axs[1, 0], 0.1)
        _demo_rbf(axs[1, 1], 500)
        print("plot ready")
    finally:
        sys.stdout = original_stdout


def _demo_linear(ax: plt.Axes, c: float) -> None:
    """Demo with linear kernel on linearly separable data."""
    x, y = make_blobs(
        n_samples=500, centers=2, n_features=2, random_state=1
    )
    y[y == 0] = -1

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = np.hstack((y.reshape(-1, 1), x))
    kernel = Kernel("linear")
    model = SmoSVM(data, kernel, cost=c, auto_norm=False)
    model.fit()
    _plot(model, data, ax)


def _demo_rbf(ax: plt.Axes, c: float) -> None:
    """Demo with RBF kernel on circular data."""
    x, y = make_circles(
        n_samples=500, noise=0.1, factor=0.1, random_state=1
    )
    y[y == 0] = -1

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = np.hstack((y.reshape(-1, 1), x))
    kernel = Kernel("rbf", gamma=0.5)
    model = SmoSVM(data, kernel, cost=c, auto_norm=False)
    model.fit()
    _plot(model, data, ax)


def _plot(model: SmoSVM, data: np.ndarray, ax: plt.Axes, res: int = 100) -> None:
    """Plot decision boundary and data points."""
    x = data[:, 1]
    y = data[:, 2]
    labels = data[:, 0]

    x_range = np.linspace(x.min(), x.max(), res)
    y_range = np.linspace(y.min(), y.max(), res)

    grid_points = np.array([[a, b] for a in x_range for b in y_range])
    predictions = model.predict(grid_points, classify=False)
    grid = predictions.reshape((len(x_range), len(y_range)))

    ax.contour(x_range, y_range, grid.T, levels=[-1, 0, 1], colors=["blue", "black", "red"])
    ax.scatter(x, y, c=labels, cmap=plt.cm.Dark2, alpha=0.5, s=20)

    support_idx = model.support
    ax.scatter(
        x[support_idx],
        y[support_idx],
        c=labels[support_idx],
        cmap=plt.cm.Dark2,
        edgecolors="k",
        s=60,
        linewidths=1.5,
    )


if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
