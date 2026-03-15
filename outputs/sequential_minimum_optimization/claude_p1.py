"""
Support Vector Machine implementation using the Simplified Sequential Minimal Optimization (SMO) algorithm.

This module provides:
- SmoSVM: A binary SVM classifier trained via a simplified SMO algorithm.
- Kernel: A callable kernel function supporting linear, polynomial, and RBF kernels.
- Demonstrations on the UCI Breast Cancer Wisconsin dataset and synthetic datasets.
"""

import os
import sys
import urllib.request
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

# URL for the UCI Breast Cancer Wisconsin (Diagnostic) dataset
CANCER_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
CANCER_CSV_PATH = "cancer.csv"


class SmoSVM:
    """
    Binary Support Vector Machine trained using a simplified SMO algorithm.

    The training data is expected as a 2D numpy array where:
        - Column 0 contains the labels (+1 or -1).
        - Columns 1: contain the feature values.

    Attributes:
        train: The original training data array.
        kernel: Kernel function used for computing inner products.
        c: Regularization parameter (cost).
        b: Bias term.
        tol: Tolerance for KKT violation checks.
        auto_norm: Whether to apply min-max normalization to features.
        tags: Array of training labels.
        samples: Array of (possibly normalized) training features.
        alphas: Lagrange multipliers for each training sample.
        err: Error cache for unbound samples.
        kmat: Precomputed kernel matrix for training samples.
    """

    _MIN_TOLERANCE = 0.001
    _EPS = 0.001

    def __init__(
        self,
        train: np.ndarray,
        kernel_func: Callable,
        alpha_list: Optional[np.ndarray] = None,
        cost: float = 0.4,
        b: float = 0.0,
        tolerance: float = 0.001,
        auto_norm: bool = True,
    ) -> None:
        """
        Initialize the SmoSVM classifier.

        Args:
            train: Training data with labels in column 0 and features in columns 1:.
            kernel_func: A callable kernel function (e.g., an instance of Kernel).
            alpha_list: Initial Lagrange multipliers. Defaults to zeros.
            cost: Regularization parameter C.
            b: Initial bias term.
            tolerance: Tolerance for KKT condition checks.
            auto_norm: If True, apply min-max normalization to features.
        """
        self.train = train
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)
        self.tol = np.float64(max(tolerance, self._MIN_TOLERANCE))
        self.auto_norm = auto_norm

        # Extract labels and features
        self.tags = train[:, 0]
        raw_features = train[:, 1:]
        self.samples = self._normalize(raw_features) if self.auto_norm else raw_features

        # Initialize Lagrange multipliers
        self.alphas = (
            np.zeros(train.shape[0]) if alpha_list is None else alpha_list.copy()
        )

        # Error cache and sample indices
        self.err = np.zeros(len(self.samples))
        self._sample_indices = list(range(len(self.samples)))

        # Precompute kernel matrix
        self.kmat = self._build_kernel_matrix()

    def _build_kernel_matrix(self) -> np.ndarray:
        """
        Precompute the kernel matrix for all pairs of training samples.

        Returns:
            A symmetric (n x n) matrix where entry (i, j) = kernel(sample_i, sample_j).
        """
        n_samples = len(self.samples)
        matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                matrix[i][j] = self.kernel(self.samples[i], self.samples[j])

        return matrix

    def _compute_kernel(self, i: int, j: Union[int, np.ndarray]) -> float:
        """
        Compute or look up the kernel value.

        Args:
            i: Index of the first training sample.
            j: Either an index into the training set or a raw feature vector.

        Returns:
            The kernel value K(sample_i, sample_j) or K(sample_i, vector_j).
        """
        if isinstance(j, np.ndarray):
            return self.kernel(self.samples[i], j)
        return self.kmat[i][j]

    def _is_unbound(self, i: int) -> bool:
        """Check if the i-th Lagrange multiplier is strictly between 0 and C."""
        return 0 < self.alphas[i] < self.c

    def _compute_error(self, i: int) -> float:
        """
        Compute the prediction error for the i-th training sample.

        For unbound samples, returns the cached error value.
        Otherwise, computes E_i = f(x_i) - y_i.

        Args:
            i: Index of the training sample.

        Returns:
            The error value E_i.
        """
        if self._is_unbound(i):
            return self.err[i]

        prediction = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return prediction - self.tags[i]

    def _violates_kkt(self, i: int) -> bool:
        """
        Check whether the i-th sample violates the KKT conditions.

        Args:
            i: Index of the training sample.

        Returns:
            True if the sample violates KKT conditions, False otherwise.
        """
        r = self._compute_error(i) * self.tags[i]
        violates_lower = (r < -self.tol) and (self.alphas[i] < self.c)
        violates_upper = (r > self.tol) and (self.alphas[i] > 0)
        return violates_lower or violates_upper

    def _compute_bounds(
        self, a1: float, a2: float, y1: float, y2: float
    ) -> tuple:
        """
        Compute the lower and upper bounds (L, H) for the new alpha_2.

        Args:
            a1: Current alpha value for sample 1.
            a2: Current alpha value for sample 2.
            y1: Label for sample 1.
            y2: Label for sample 2.

        Returns:
            Tuple (L, H) representing the valid range for alpha_2.
        """
        if y1 * y2 == -1:
            lower = max(0, a2 - a1)
            upper = min(self.c, self.c + a2 - a1)
        else:
            lower = max(0, a1 + a2 - self.c)
            upper = min(self.c, a1 + a2)
        return lower, upper

    def _update_bias(
        self,
        e1: float,
        e2: float,
        y1: float,
        y2: float,
        a1_old: float,
        a2_old: float,
        a1_new: float,
        a2_new: float,
        k11: float,
        k12: float,
        k22: float,
    ) -> None:
        """
        Update the bias term after optimizing a pair of alphas.

        Args:
            e1, e2: Errors for the two samples before the update.
            y1, y2: Labels for the two samples.
            a1_old, a2_old: Old alpha values.
            a1_new, a2_new: New alpha values.
            k11, k12, k22: Kernel values between the two samples.
        """
        delta_a1 = a1_new - a1_old
        delta_a2 = a2_new - a2_old

        b1 = -e1 - y1 * k11 * delta_a1 - y2 * k12 * delta_a2 + self.b
        b2 = -e2 - y1 * k12 * delta_a1 - y2 * k22 * delta_a2 + self.b

        if 0 < a1_new < self.c:
            self.b = b1
        elif 0 < a2_new < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

    def fit(self) -> None:
        """
        Train the SVM using the simplified SMO algorithm.

        Iterates over all pairs of samples, optimizing Lagrange multipliers
        until no further KKT violations are found.
        """
        changed = True

        while changed:
            changed = False

            for i1 in self._sample_indices:
                if not self._violates_kkt(i1):
                    continue

                for i2 in self._sample_indices:
                    if i1 == i2:
                        continue

                    y1, y2 = self.tags[i1], self.tags[i2]
                    a1, a2 = self.alphas[i1], self.alphas[i2]
                    e1, e2 = self._compute_error(i1), self._compute_error(i2)

                    # Compute bounds for alpha_2
                    lower, upper = self._compute_bounds(a1, a2, y1, y2)
                    if lower == upper:
                        continue

                    # Compute second derivative (eta)
                    k11 = self._compute_kernel(i1, i1)
                    k22 = self._compute_kernel(i2, i2)
                    k12 = self._compute_kernel(i1, i2)
                    eta = k11 + k22 - 2 * k12

                    if eta <= 0:
                        continue

                    # Update alpha_2 and clip to bounds
                    a2_new = a2 + (y2 * (e1 - e2)) / eta
                    a2_new = np.clip(a2_new, lower, upper)

                    # Update alpha_1 using the constraint
                    a1_new = a1 + y1 * y2 * (a2 - a2_new)

                    # Store new alpha values
                    self.alphas[i1] = a1_new
                    self.alphas[i2] = a2_new

                    # Update bias
                    self._update_bias(e1, e2, y1, y2, a1, a2, a1_new, a2_new, k11, k12, k22)

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """
        Predict labels or decision values for test samples.

        Args:
            test: Feature matrix of shape (n_samples, n_features).
            classify: If True, return class labels (+1/-1).
                      If False, return raw decision values.

        Returns:
            Array of predictions.
        """
        if self.auto_norm:
            test = self._normalize(test)

        predictions = []
        for sample in test:
            decision_value = sum(
                self.alphas[i] * self.tags[i] * self._compute_kernel(i, sample)
                for i in range(len(self.samples))
            )
            decision_value += self.b

            if classify:
                predictions.append(1 if decision_value > 0 else -1)
            else:
                predictions.append(decision_value)

        return np.array(predictions)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to feature data.

        On the first call, computes and stores the min/max values
        from the provided data (expected to be training data).

        Args:
            data: Feature array to normalize.

        Returns:
            Normalized feature array with values in [0, 1].
        """
        if not hasattr(self, "_data_min"):
            self._data_min = np.min(data, axis=0)
            self._data_max = np.max(data, axis=0)

        return (data - self._data_min) / (self._data_max - self._data_min)

    @property
    def support(self) -> List[int]:
        """
        Get indices of support vectors (samples with alpha > 0).

        Returns:
            List of indices corresponding to support vectors.
        """
        return [i for i, alpha in enumerate(self.alphas) if alpha > 0]


class Kernel:
    """
    Callable kernel function for SVM.

    Supports linear, polynomial, and radial basis function (RBF) kernels.

    Attributes:
        name: Kernel type ('linear', 'poly', or 'rbf').
        degree: Degree for polynomial kernel.
        coef0: Independent coefficient for linear and polynomial kernels.
        gamma: Scale parameter for polynomial and RBF kernels.
    """

    VALID_KERNELS = {"linear", "poly", "rbf"}

    def __init__(
        self,
        kernel: str,
        degree: float = 1.0,
        coef0: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        """
        Initialize the Kernel function.

        Args:
            kernel: Type of kernel ('linear', 'poly', or 'rbf').
            degree: Degree of the polynomial kernel.
            coef0: Independent coefficient in linear and polynomial kernels.
            gamma: Scale parameter for polynomial and RBF kernels.
        """
        self.name = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute the kernel value between two vectors.

        Args:
            v1: First input vector.
            v2: Second input vector.

        Returns:
            The kernel evaluation K(v1, v2).
        """
        if self.name == "linear":
            return np.inner(v1, v2) + self.coef0

        if self.name == "poly":
            return (self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree

        if self.name == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(v1 - v2) ** 2)

        # Default fallback: linear kernel without bias
        return np.inner(v1, v2)


def _download_cancer_data(url: str, filepath: str) -> None:
    """
    Download the breast cancer dataset from UCI and save it locally.

    Args:
        url: URL of the dataset.
        filepath: Local path to save the downloaded CSV file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla"})
    with urllib.request.urlopen(request) as response:
        data = response.read().decode()

    with open(filepath, "w") as file:
        file.write(data)


def _load_cancer_data(filepath: str) -> np.ndarray:
    """
    Load and preprocess the breast cancer dataset from a CSV file.

    Drops the ID column, removes rows with missing values,
    and maps diagnosis labels ('M' -> +1, 'B' -> -1).

    Args:
        filepath: Path to the CSV file.

    Returns:
        Numpy array with labels in column 0 and features in columns 1:.
    """
    df = pd.read_csv(filepath, header=None, dtype={0: str})

    # Drop the patient ID column
    del df[df.columns[0]]

    df = df.dropna(axis=0)

    # Convert diagnosis labels to numeric values
    df = df.replace({"M": 1.0, "B": -1.0})

    return np.array(df)


def test_cancer() -> None:
    """
    Train and evaluate the SVM on the UCI Breast Cancer Wisconsin dataset.

    Downloads the data if not already cached locally, trains an RBF-kernel
    SVM using SMO, and prints the test accuracy.
    """
    # Download data if needed
    if not os.path.exists(CANCER_CSV_PATH):
        _download_cancer_data(CANCER_DATA_URL, CANCER_CSV_PATH)

    # Load and split data
    data = _load_cancer_data(CANCER_CSV_PATH)
    train_split = 328

    train_data = data[:train_split, :]
    test_data = data[train_split:, :]

    test_labels = test_data[:, 0]
    test_features = test_data[:, 1:]

    # Configure and train the model
    kernel = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    initial_alphas = np.zeros(train_data.shape[0])
    model = SmoSVM(train_data, kernel, initial_alphas)
    model.fit()

    # Evaluate accuracy
    predictions = model.predict(test_features)
    accuracy = np.mean(predictions == test_labels)
    print("accuracy", accuracy)


def demo() -> None:
    """
    Generate a 2x2 grid of demo plots showing SVM decision boundaries.

    Demonstrates linear and RBF kernels with low (0.1) and high (500) cost values.
    """
    # Suppress stdout during model training to avoid clutter
    sys.stdout = open(os.devnull, "w")

    fig, axes = plt.subplots(2, 2)

    _demo_linear(axes[0][0], cost=0.1)
    _demo_linear(axes[0][1], cost=500)
    _demo_rbf(axes[1][0], cost=0.1)
    _demo_rbf(axes[1][1], cost=500)

    # Restore stdout
    sys.stdout = sys.__stdout__
    print("plot ready")


def _prepare_blob_data(
    n_samples: int = 500, random_state: int = 1
) -> np.ndarray:
    """
    Generate and standardize linearly separable blob data.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        Array with labels in column 0 and standardized features in columns 1:.
    """
    features, labels = make_blobs(
        n_samples=n_samples, centers=2, n_features=2, random_state=random_state
    )
    labels[labels == 0] = -1

    scaler = StandardScaler()
    features = scaler.fit_transform(features, labels)

    return np.hstack((labels.reshape(n_samples, 1), features))


def _prepare_circle_data(
    n_samples: int = 500,
    noise: float = 0.1,
    factor: float = 0.1,
    random_state: int = 1,
) -> np.ndarray:
    """
    Generate and standardize non-linearly separable circle data.

    Args:
        n_samples: Number of samples to generate.
        noise: Standard deviation of Gaussian noise added to the data.
        factor: Scale factor between inner and outer circle.
        random_state: Random seed for reproducibility.

    Returns:
        Array with labels in column 0 and standardized features in columns 1:.
    """
    features, labels = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=random_state
    )
    labels[labels == 0] = -1

    scaler = StandardScaler()
    features = scaler.fit_transform(features, labels)

    return np.hstack((labels.reshape(n_samples, 1), features))


def _demo_linear(ax: plt.Axes, cost: float) -> None:
    """
    Demo SVM with a linear kernel on linearly separable blob data.

    Args:
        ax: Matplotlib axes to plot on.
        cost: Regularization parameter C.
    """
    data = _prepare_blob_data()
    kernel = Kernel("linear")
    model = SmoSVM(data, kernel, cost=cost, auto_norm=False)
    model.fit()
    _plot_decision_boundary(model, data, ax)


def _demo_rbf(ax: plt.Axes, cost: float) -> None:
    """
    Demo SVM with an RBF kernel on non-linearly separable circle data.

    Args:
        ax: Matplotlib axes to plot on.
        cost: Regularization parameter C.
    """
    data = _prepare_circle_data()
    kernel = Kernel("rbf")
    model = SmoSVM(data, kernel, cost=cost, auto_norm=False)
    model.fit()
    _plot_decision_boundary(model, data, ax)


def _plot_decision_boundary(
    model: SmoSVM,
    data: np.ndarray,
    ax: plt.Axes,
    resolution: int = 100,
) -> None:
    """
    Plot the SVM decision boundary, data points, and support vectors.

    Args:
        model: Trained SmoSVM model.
        data: Training data with labels in column 0.
        ax: Matplotlib axes to plot on.
        resolution: Number of grid points per axis for the contour plot.
    """
    feature_x = data[:, 1]
    feature_y = data[:, 2]
    labels = data[:, 0]

    # Create evaluation grid
    x_range = np.linspace(feature_x.min(), feature_x.max(), resolution)
    y_range = np.linspace(feature_y.min(), feature_y.max(), resolution)
    grid_points = np.array(
        [(a, b) for a in x_range for b in y_range]
    ).reshape(resolution * resolution, 2)

    # Get decision values on the grid
    decision_values = model.predict(grid_points, classify=False)
    decision_grid = decision_values.reshape((len(x_range), len(y_range)))

    # Plot contours for decision boundary and margins
    ax.contour(x_range, y_range, np.asmatrix(decision_grid).T, levels=(-1, 0, 1))

    # Plot all data points
    ax.scatter(feature_x, feature_y, c=labels, cmap=plt.cm.Dark2, alpha=0.5)

    # Highlight support vectors
    support_indices = model.support
    ax.scatter(
        feature_x[support_indices],
        feature_y[support_indices],
        c=labels[support_indices],
        cmap=plt.cm.Dark2,
    )


if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
