import os
import sys
import urllib.request
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

# Constants
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
DATA_FILENAME = "cancer.csv"


class Kernel:
    """
    A unified kernel class that computes the inner product mapping
    for 'linear', 'poly', and 'rbf' kernels.
    """

    def __init__(
        self,
        kernel: str,
        degree: float = 1.0,
        coef0: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        """
        Initialize the kernel function.

        Args:
            kernel: The type of kernel ('linear', 'poly', 'rbf').
            degree: Degree of the polynomial kernel.
            coef0: Independent term in poly and linear kernels.
            gamma: Kernel coefficient for rbf and poly.
        """
        self.name = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute the kernel evaluation between two vectors.

        Args:
            v1: First feature vector.
            v2: Second feature vector.

        Returns:
            The computed kernel value.
        """
        if self.name == "linear":
            return float(np.inner(v1, v2) + self.coef0)

        if self.name == "poly":
            return float((self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree)

        if self.name == "rbf":
            return float(np.exp(-1.0 * (self.gamma * np.linalg.norm(v1 - v2) ** 2)))

        return float(np.inner(v1, v2))


class SmoSVM:
    """
    Support Vector Machine classifier trained using a simplified 
    Sequential Minimal Optimization (SMO) algorithm.
    """

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
        """
        Initialize the SMO SVM model.

        Args:
            train: Training data where the first column is the target tag (+1 or -1).
            kernel_func: An instance of the Kernel class.
            alpha_list: Initial Lagrange multipliers. Defaults to zeros.
            cost: Regularization parameter (C).
            b: Initial bias term.
            tolerance: Tolerance for checking the KKT conditions.
            auto_norm: Whether to min-max normalize features automatically.
        """
        self.train = train
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)
        self.tol = np.float64(tolerance)

        if tolerance <= 0.0001:
            self.tol = np.float64(0.001)

        self.auto_norm = auto_norm
        self.tags = train[:, 0]

        if self.auto_norm:
            self.samples = self._norm(train[:, 1:])
        else:
            self.samples = train[:, 1:]

        if alpha_list is None:
            self.alphas = np.zeros(train.shape[0])
        else:
            self.alphas = alpha_list

        self.err = np.zeros(len(self.samples))
        self._eps = 0.001

        self.indexes = list(range(len(self.samples)))
        self.kmat = self._build_k()
        self.unbound: List[int] = []

    def _build_k(self) -> np.ndarray:
        """Precompute the kernel matrix for all training samples."""
        n = len(self.samples)
        m = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                m[i][j] = self.kernel(self.samples[i], self.samples[j])

        return m

    def _k(self, i: int, j: Union[int, np.ndarray]) -> float:
        """
        Retrieve or compute the kernel between sample i and element j.
        If j is an index, use the precomputed matrix.
        """
        if isinstance(j, np.ndarray):
            return self.kernel(self.samples[i], j)
        return self.kmat[i][j]

    def _is_unbound(self, i: int) -> bool:
        """Check if the alpha value is strictly between 0 and C."""
        return 0 < self.alphas[i] < self.c

    def _e(self, i: int) -> float:
        """
        Compute the prediction error for the i-th training sample.
        Note: The cached `self.err` is returned if the sample is unbound.
        """
        if self._is_unbound(i):
            return float(self.err[i])

        gx = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return float(gx - self.tags[i])

    def _check_kkt(self, i: int) -> bool:
        """Check if the i-th sample violates the KKT conditions."""
        r = self._e(i) * self.tags[i]

        if (r < -self.tol and self.alphas[i] < self.c) or (r > self.tol and self.alphas[i] > 0):
            return True
        return False

    def fit(self) -> None:
        """Train the SVM model using the SMO heuristic."""
        changed = True

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

                    # Compute boundaries for alpha
                    if s == -1:
                        L = max(0.0, a2 - a1)
                        H = min(self.c, self.c + a2 - a1)
                    else:
                        L = max(0.0, a1 + a2 - self.c)
                        H = min(self.c, a1 + a2)

                    if L == H:
                        continue

                    k11 = self._k(i1, i1)
                    k22 = self._k(i2, i2)
                    k12 = self._k(i1, i2)

                    eta = k11 + k22 - 2 * k12

                    if eta <= 0:
                        continue

                    # Update alpha2
                    a2new = a2 + (y2 * (e1 - e2)) / eta

                    # Clip alpha2 to boundaries
                    if a2new > H:
                        a2new = H
                    elif a2new < L:
                        a2new = L

                    # Update alpha1
                    a1new = a1 + s * (a2 - a2new)

                    self.alphas[i1] = a1new
                    self.alphas[i2] = a2new

                    # Update bias term
                    b1 = -e1 - y1 * k11 * (a1new - a1) - y2 * k12 * (a2new - a2) + self.b
                    b2 = -e2 - y1 * k12 * (a1new - a1) - y2 * k22 * (a2new - a2) + self.b

                    if 0 < a1new < self.c:
                        self.b = b1
                    elif 0 < a2new < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """
        Predict outcomes for new test samples.

        Args:
            test: Test dataset features.
            classify: If True, outputs class labels (+1, -1). If False, outputs raw predictions.

        Returns:
            An array of predictions.
        """
        if self.auto_norm:
            test = self._norm(test)

        out = []

        for sample in test:
            v = 0.0

            for i in range(len(self.samples)):
                v += self.alphas[i] * self.tags[i] * self._k(i, sample)

            v += self.b

            if classify:
                out.append(1 if v > 0 else -1)
            else:
                out.append(v)

        return np.array(out)

    def _norm(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to the data based on training parameters."""
        if not hasattr(self, "min"):
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)

        return (data - self.min) / (self.max - self.min)

    @property
    def support(self) -> List[int]:
        """Get the indices of the support vectors."""
        res = []

        for i in range(len(self.alphas)):
            if self.alphas[i] > 0:
                res.append(i)

        return res


def test_cancer() -> None:
    """
    Download the Breast Cancer dataset, train the SVM, 
    and output the accuracy of the model on the test set.
    """
    if not os.path.exists(DATA_FILENAME):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla"})
        with urllib.request.urlopen(req) as response:
            data = response.read().decode()

        with open(DATA_FILENAME, "w", encoding="utf-8") as f:
            f.write(data)

    df = pd.read_csv(DATA_FILENAME, header=None, dtype={0: str})
    
    # Remove the ID column
    del df[df.columns[0]]

    df = df.dropna(axis=0)
    df = df.replace({"M": 1.0, "B": -1.0})

    arr = np.array(df)

    train = arr[:328, :]
    test = arr[328:, :]

    test_tags = test[:, 0]
    test_x = test[:, 1:]

    ker = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    alphas = np.zeros(train.shape[0])

    model = SmoSVM(train, ker, alphas)
    model.fit()

    pred = model.predict(test_x)

    good = sum(1 for p, t in zip(pred, test_tags) if p == t)

    print("accuracy", good / len(pred))


def demo() -> None:
    """Run visual demonstrations of the SMO SVM with different kernels."""
    original_stdout = sys.stdout
    # Suppress console output during plotting
    sys.stdout = open(os.devnull, "w")

    try:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        _demo_linear(axs[0, 0], 0.1)
        _demo_linear(axs[0, 1], 500)
        _demo_rbf(axs[1, 0], 0.1)
        _demo_rbf(axs[1, 1], 500)
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    print("plot ready")


def _demo_linear(ax: plt.Axes, cost: float) -> None:
    """Plot the decision boundary of a linear SVM."""
    x, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=1)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)

    data = np.hstack((y.reshape(-1, 1), x))

    k = Kernel("linear")
    m = SmoSVM(data, k, cost=cost, auto_norm=False)
    m.fit()

    _plot(m, data, ax)


def _demo_rbf(ax: plt.Axes, cost: float) -> None:
    """Plot the decision boundary of an RBF SVM."""
    x, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)

    data = np.hstack((y.reshape(-1, 1), x))

    k = Kernel("rbf")
    m = SmoSVM(data, k, cost=cost, auto_norm=False)
    m.fit()

    _plot(m, data, ax)


def _plot(model: SmoSVM, data: np.ndarray, ax: plt.Axes, res: int = 100) -> None:
    """Helper function to map the SVM decision plane and scatter the datasets."""
    x = data[:, 1]
    y = data[:, 2]
    tags = data[:, 0]

    xr = np.linspace(x.min(), x.max(), res)
    yr = np.linspace(y.min(), y.max(), res)

    pts = np.array([(a, b) for a in xr for b in yr]).reshape(res * res, 2)
    pred = model.predict(pts, classify=False)

    grid = pred.reshape((len(xr), len(yr)))

    ax.contour(xr, yr, grid.T, levels=[-1, 0, 1])
    ax.scatter(x, y, c=tags, cmap=plt.cm.Dark2, alpha=0.5)

    sup = model.support
    ax.scatter(x[sup], y[sup], c=tags[sup], cmap=plt.cm.Dark2)


if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
