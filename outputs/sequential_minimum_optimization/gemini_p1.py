"""
Support Vector Machine implementation using a simplified Sequential Minimal Optimization (SMO) algorithm.
Provides a binary SVM classifier, various kernel functions, and demonstrations.
"""

import os
import sys
import urllib.request
from contextlib import redirect_stdout
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

# Constants
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


class SmoSVM:
    """
    Binary Support Vector Machine using a simplified SMO algorithm.
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
        """
        Initialize the SMO SVM model.

        Args:
            train (np.ndarray): Training dataset where the first column is the label and the rest are features.
            kernel_func (Callable): Kernel function to use for mapping features.
            alpha_list (Optional[np.ndarray]): Initial Lagrange multipliers. Defaults to zeros.
            cost (float): Regularization parameter C. Defaults to 0.4.
            b (float): Initial bias. Defaults to 0.0.
            tolerance (float): Tolerance for KKT condition checks. Defaults to 0.001.
            auto_norm (bool): If True, apply min-max normalization to the features. Defaults to True.
        """
        self.train = train
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)
        self.tol = np.float64(0.001 if tolerance <= 0.0001 else tolerance)
        self.auto_norm = auto_norm

        # Extract labels
        self.tags = train[:, 0]

        # Normalize or extract features
        features = train[:, 1:]
        self.samples = self._norm(features) if self.auto_norm else features

        # Initialize Lagrange multipliers
        if alpha_list is None:
            self.alphas = np.zeros(train.shape[0])
        else:
            self.alphas = alpha_list

        # Error cache (Note: kept for structural equivalence with the original code)
        self.err = np.zeros(len(self.samples))

        self.indexes = list(range(len(self.samples)))
        self.kmat = self._build_k()

    def _build_k(self) -> np.ndarray:
        """Precompute the kernel matrix for all training samples."""
        n_samples = len(self.samples)
        matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                matrix[i, j] = self.kernel(self.samples[i], self.samples[j])

        return matrix

    def _k(self, i: int, j: Union[int, np.ndarray]) -> float:
        """Look up or compute the kernel between the i-th sample and j-th sample/array."""
        if isinstance(j, np.ndarray):
            return self.kernel(self.samples[i], j)
        return self.kmat[i, j]

    def _is_unbound(self, i: int) -> bool:
        """Check if the alpha value is strictly between 0 and C (unbound)."""
        return 0 < self.alphas[i] < self.c

    def _e(self, i: int) -> float:
        """Compute the prediction error for the i-th sample."""
        if self._is_unbound(i):
            return self.err[i]

        prediction = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return prediction - self.tags[i]

    def _check_kkt(self, i: int) -> bool:
        """Check if the i-th sample violates the Karush-Kuhn-Tucker (KKT) conditions."""
        margin_error = self._e(i) * self.tags[i]
        violates_lower_bound = (margin_error < -self.tol) and (self.alphas[i] < self.c)
        violates_upper_bound = (margin_error > self.tol) and (self.alphas[i] > 0)

        return violates_lower_bound or violates_upper_bound

    def fit(self) -> None:
        """Train the SVM using the simplified SMO optimization loop."""
        changed = True

        while changed:
            changed = False

            for i1 in self.indexes:
                if not self._check_kkt(i1):
                    continue

                for i2 in self.indexes:
                    if i1 == i2:
                        continue

                    label1, label2 = self.tags[i1], self.tags[i2]
                    alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
                    error1, error2 = self._e(i1), self._e(i2)

                    label_product = label1 * label2

                    # Compute bounds L and H for alpha2
                    if label_product == -1:
                        low_bound = max(0.0, alpha2 - alpha1)
                        high_bound = min(self.c, self.c + alpha2 - alpha1)
                    else:
                        low_bound = max(0.0, alpha1 + alpha2 - self.c)
                        high_bound = min(self.c, alpha1 + alpha2)

                    if low_bound == high_bound:
                        continue

                    # Compute eta (second derivative of the objective function)
                    k11 = self._k(i1, i1)
                    k22 = self._k(i2, i2)
                    k12 = self._k(i1, i2)
                    eta = k11 + k22 - 2 * k12

                    if eta <= 0:
                        continue

                    # Update alpha2 and clip it to bounds
                    alpha2_new = alpha2 + (label2 * (error1 - error2)) / eta
                    if alpha2_new > high_bound:
                        alpha2_new = high_bound
                    elif alpha2_new < low_bound:
                        alpha2_new = low_bound

                    # Update alpha1
                    alpha1_new = alpha1 + label_product * (alpha2 - alpha2_new)

                    self.alphas[i1] = alpha1_new
                    self.alphas[i2] = alpha2_new

                    # Update bias
                    b1 = -error1 - label1 * k11 * (alpha1_new - alpha1) - label2 * k12 * (alpha2_new - alpha2) + self.b
                    b2 = -error2 - label1 * k12 * (alpha1_new - alpha1) - label2 * k22 * (alpha2_new - alpha2) + self.b

                    if 0 < alpha1_new < self.c:
                        self.b = b1
                    elif 0 < alpha2_new < self.c:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """
        Predict outputs for a given test dataset.

        Args:
            test (np.ndarray): The test features.
            classify (bool): If True, returns discrete labels (-1 or 1). If False, returns decision values.

        Returns:
            np.ndarray: Predicted labels or decision values.
        """
        if self.auto_norm:
            test = self._norm(test)

        predictions = []

        for sample in test:
            decision_value = sum(
                self.alphas[i] * self.tags[i] * self._k(i, sample)
                for i in range(len(self.samples))
            ) + self.b

            if classify:
                predictions.append(1 if decision_value > 0 else -1)
            else:
                predictions.append(decision_value)

        return np.array(predictions)

    def _norm(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalize the dataset based on the training data bounds."""
        if not hasattr(self, "data_min"):
            self.data_min = np.min(data, axis=0)
            self.data_max = np.max(data, axis=0)

        return (data - self.data_min) / (self.data_max - self.data_min)

    @property
    def support(self) -> List[int]:
        """Return the indices of the support vectors."""
        return [i for i, alpha in enumerate(self.alphas) if alpha > 0]


class Kernel:
    """Callable Kernel function supporting linear, polynomial, and RBF mappings."""

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
            return float(np.exp(-1 * (self.gamma * np.linalg.norm(v1 - v2) ** 2)))

        # Default fallback to inner product
        return float(np.inner(v1, v2))


def test_cancer() -> None:
    """Download the breast cancer dataset, train the SVM, and print test accuracy."""
    filename = "cancer.csv"

    if not os.path.exists(filename):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla"})
        with urllib.request.urlopen(req) as response:
            data = response.read().decode()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)

    df = pd.read_csv(filename, header=None, dtype={0: str})
    
    # Drop ID column
    df = df.drop(df.columns[0], axis=1)
    df = df.dropna(axis=0)

    # Convert labels to 1 and -1
    df = df.replace({"M": 1.0, "B": -1.0})

    arr = np.array(df)

    # Simple train/test split
    train = arr[:328, :]
    test = arr[328:, :]

    test_tags = test[:, 0]
    test_features = test[:, 1:]

    ker = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    initial_alphas = np.zeros(train.shape[0])

    model = SmoSVM(train, ker, initial_alphas)
    model.fit()

    predictions = model.predict(test_features)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_tags)
    print("accuracy", accuracy)


def demo() -> None:
    """Display a 2x2 grid of decision boundaries using synthetic datasets."""
    # Suppress console output during training
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        fig, axs = plt.subplots(2, 2)

        _demo_linear(axs[0, 0], 0.1)
        _demo_linear(axs[0, 1], 500)
        _demo_rbf(axs[1, 0], 0.1)
        _demo_rbf(axs[1, 1], 500)

    print("plot ready")


def _demo_linear(ax: plt.Axes, c: float) -> None:
    """Train and plot an SVM with a linear kernel on synthetic blobs."""
    x, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=1)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)
    data = np.hstack((y.reshape(500, 1), x))

    k = Kernel("linear")
    model = SmoSVM(data, k, cost=c, auto_norm=False)
    model.fit()

    _plot(model, data, ax)


def _demo_rbf(ax: plt.Axes, c: float) -> None:
    """Train and plot an SVM with an RBF kernel on synthetic circles."""
    x, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
    y[y == 0] = -1

    sc = StandardScaler()
    x = sc.fit_transform(x, y)
    data = np.hstack((y.reshape(500, 1), x))

    k = Kernel("rbf")
    model = SmoSVM(data, k, cost=c, auto_norm=False)
    model.fit()

    _plot(model, data, ax)


def _plot(model: SmoSVM, data: np.ndarray, ax: plt.Axes, res: int = 100) -> None:
    """Plot the data, SVM decision boundaries, and support vectors."""
    features_x = data[:, 1]
    features_y = data[:, 2]
    tags = data[:, 0]

    xr = np.linspace(features_x.min(), features_x.max(), res)
    yr = np.linspace(features_y.min(), features_y.max(), res)

    # Generate a grid of points over the coordinate ranges
    pts = np.array([(a, b) for a in xr for b in yr]).reshape(res * res, 2)

    pred = model.predict(pts, classify=False)
    grid = pred.reshape((len(xr), len(yr)))

    # Plot the decision boundaries
    ax.contour(xr, yr, grid.T, levels=(-1, 0, 1))

    # Scatter plot of original dataset
    ax.scatter(features_x, features_y, c=tags, cmap=plt.cm.Dark2, alpha=0.5)

    # Scatter plot highlighting support vectors
    sup = model.support
    ax.scatter(features_x[sup], features_y[sup], c=tags[sup], cmap=plt.cm.Dark2)


if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
