"""
Support Vector Machine implementation using a simplified Sequential Minimal
Optimization (SMO) algorithm.

Includes:
- SmoSVM: SVM classifier trained via simplified SMO.
- Kernel: Callable kernel function (linear, polynomial, RBF).
- Demos and tests on synthetic data and the UCI breast cancer dataset.
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

# UCI Breast Cancer Wisconsin (Diagnostic) dataset URL
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
CANCER_CSV_PATH = "cancer.csv"

# Number of training samples used in the cancer dataset split
CANCER_TRAIN_SIZE = 328


class Kernel:
    """Callable kernel function supporting linear, polynomial, and RBF types.

    Attributes:
        name: Kernel type identifier ('linear', 'poly', or 'rbf').
        degree: Polynomial degree (used only for 'poly').
        coef0: Independent coefficient (used for 'linear' and 'poly').
        gamma: Kernel coefficient (used for 'poly' and 'rbf').
    """

    def __init__(
        self,
        kernel: str,
        degree: float = 1.0,
        coef0: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        """Initialize the kernel function.

        Args:
            kernel: Type of kernel ('linear', 'poly', 'rbf').
            degree: Degree for polynomial kernel.
            coef0: Independent coefficient in the kernel function.
            gamma: Kernel coefficient for 'poly' and 'rbf'.
        """
        self.name = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute the kernel value between two feature vectors.

        Args:
            v1: First feature vector.
            v2: Second feature vector.

        Returns:
            Scalar kernel value.
        """
        if self.name == "linear":
            return np.inner(v1, v2) + self.coef0

        if self.name == "poly":
            return (self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree

        if self.name == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(v1 - v2) ** 2)

        # Fallback: plain inner product
        return np.inner(v1, v2)


class SmoSVM:
    """Support Vector Machine trained with a simplified SMO algorithm.

    The training data is expected as an array where the first column contains
    labels (+1 / -1) and remaining columns are features.

    Attributes:
        kernel: Kernel function used for computing similarities.
        c: Regularization (cost) parameter.
        b: Bias term.
        tol: Numerical tolerance for KKT violation checks.
        auto_norm: Whether to apply min-max normalization to features.
        tags: Label vector from the training data.
        samples: Feature matrix (possibly normalized).
        alphas: Lagrange multipliers.
        err: Error cache for unbound support vectors.
        kmat: Pre-computed kernel matrix for training samples.
    """

    # Minimum acceptable tolerance value
    _MIN_TOLERANCE = 0.001

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
            train: Training data of shape (n_samples, 1 + n_features).
                   Column 0 holds the labels; remaining columns hold features.
            kernel_func: Kernel callable (e.g., a ``Kernel`` instance).
            alpha_list: Optional initial Lagrange multipliers.
            cost: Regularization parameter C.
            b: Initial bias.
            tolerance: KKT violation tolerance (clamped to >= 0.001).
            auto_norm: If True, apply min-max normalization to features.
        """
        self.kernel = kernel_func
        self.c = np.float64(cost)
        self.b = np.float64(b)
        self.tol = np.float64(max(tolerance, self._MIN_TOLERANCE))
        self.auto_norm = auto_norm

        # Separate labels and features
        self.tags: np.ndarray = train[:, 0]
        self.samples: np.ndarray = (
            self._normalize(train[:, 1:]) if self.auto_norm else train[:, 1:]
        )

        # Initialize Lagrange multipliers
        self.alphas: np.ndarray = (
            np.zeros(train.shape[0]) if alpha_list is None else alpha_list
        )

        # Error cache and sample indices
        self.err: np.ndarray = np.zeros(len(self.samples))
        self._sample_indices: List[int] = list(range(len(self.samples)))

        # Pre-compute kernel matrix for all training pairs
        self.kmat: np.ndarray = self._build_kernel_matrix()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kernel_matrix(self) -> np.ndarray:
        """Compute the full kernel matrix K[i, j] = k(x_i, x_j)."""
        n_samples = len(self.samples)
        matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                matrix[i][j] = self.kernel(self.samples[i], self.samples[j])
        return matrix

    def _compute_kernel(self, i: int, j: Union[int, np.ndarray]) -> float:
        """Return kernel value between training sample *i* and *j*.

        Args:
            i: Index of a training sample.
            j: Either another training-sample index or an arbitrary feature vector.
        """
        if isinstance(j, np.ndarray):
            return self.kernel(self.samples[i], j)
        return self.kmat[i][j]

    def _is_unbound(self, i: int) -> bool:
        """Check whether alpha_i lies strictly inside (0, C)."""
        return 0 < self.alphas[i] < self.c

    def _compute_error(self, i: int) -> float:
        """Compute (or retrieve cached) error E_i = f(x_i) - y_i."""
        if self._is_unbound(i):
            return self.err[i]

        decision_value = np.dot(self.alphas * self.tags, self.kmat[:, i]) + self.b
        return decision_value - self.tags[i]

    def _violates_kkt(self, i: int) -> bool:
        """Return True if sample *i* violates the KKT conditions."""
        r = self._compute_error(i) * self.tags[i]
        too_small = r < -self.tol and self.alphas[i] < self.c
        too_large = r > self.tol and self.alphas[i] > 0
        return too_small or too_large

    @staticmethod
    def _compute_bounds(
        y1: float, y2: float, a1: float, a2: float, c: float
    ) -> tuple:
        """Compute the lower (L) and upper (H) bounds for alpha_2.

        Args:
            y1: Label of sample 1.
            y2: Label of sample 2.
            a1: Current alpha of sample 1.
            a2: Current alpha of sample 2.
            c: Regularization parameter C.

        Returns:
            Tuple (L, H).
        """
        if y1 * y2 == -1:
            lower = max(0.0, a2 - a1)
            upper = min(c, c + a2 - a1)
        else:
            lower = max(0.0, a1 + a2 - c)
            upper = min(c, a1 + a2)
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
        """Update bias term after a successful alpha-pair update."""
        delta_a1 = a1_new - a1_old
        delta_a2 = a2_new - a2_old

        b1 = -e1 - y1 * k11 * delta_a1 - y2 * k12 * delta_a2 + self.b
        b2 = -e2 - y1 * k12 * delta_a1 - y2 * k22 * delta_a2 + self.b

        if 0 < a1_new < self.c:
            self.b = b1
        elif 0 < a2_new < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Train the SVM using a simplified SMO algorithm.

        Iterates over all sample pairs, updating Lagrange multipliers
        until no KKT-violating pair yields a productive step.
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

                    # Compute feasible region for alpha_2
                    lower, upper = self._compute_bounds(y1, y2, a1, a2, self.c)
                    if lower == upper:
                        continue

                    # Second-order derivative of the objective along the constraint
                    k11 = self._compute_kernel(i1, i1)
                    k22 = self._compute_kernel(i2, i2)
                    k12 = self._compute_kernel(i1, i2)
                    eta = k11 + k22 - 2.0 * k12
                    if eta <= 0:
                        continue

                    # Update alpha_2 and clip to [L, H]
                    a2_new = a2 + (y2 * (e1 - e2)) / eta
                    a2_new = np.clip(a2_new, lower, upper)

                    # Update alpha_1 to satisfy the linear constraint
                    a1_new = a1 + y1 * y2 * (a2 - a2_new)

                    # Store new alphas
                    self.alphas[i1] = a1_new
                    self.alphas[i2] = a2_new

                    # Update bias
                    self._update_bias(e1, e2, y1, y2, a1, a2, a1_new, a2_new, k11, k12, k22)

                    changed = True

    def predict(self, test: np.ndarray, classify: bool = True) -> np.ndarray:
        """Predict labels or decision values for new samples.

        Args:
            test: Feature matrix of shape (n_samples, n_features).
            classify: If True return class labels (+1/-1), otherwise raw
                      decision values.

        Returns:
            Array of predictions.
        """
        if self.auto_norm:
            test = self._normalize(test)

        predictions: List[float] = []

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
        """Apply min-max normalization.

        Statistics (min/max) are computed and stored on the first call
        (expected to be on training data) and reused for subsequent calls.

        Args:
            data: Feature matrix.

        Returns:
            Normalized feature matrix scaled to [0, 1].
        """
        if not hasattr(self, "_feat_min"):
            self._feat_min = np.min(data, axis=0)
            self._feat_max = np.max(data, axis=0)

        return (data - self._feat_min) / (self._feat_max - self._feat_min)

    @property
    def support(self) -> List[int]:
        """Indices of support vectors (samples with alpha > 0)."""
        return [i for i, alpha in enumerate(self.alphas) if alpha > 0]


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------


def _download_cancer_data(filepath: str = CANCER_CSV_PATH) -> None:
    """Download the WDBC cancer dataset if it is not already cached locally."""
    if os.path.exists(filepath):
        return

    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla"})
    with urllib.request.urlopen(request) as response:
        raw_data = response.read().decode()

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(raw_data)


def _load_cancer_data(
    filepath: str = CANCER_CSV_PATH,
) -> tuple:
    """Load and preprocess the WDBC dataset.

    Returns:
        Tuple of (train_array, test_labels, test_features).
    """
    _download_cancer_data(filepath)

    df = pd.read_csv(filepath, header=None, dtype={0: str})

    # Drop the patient ID column
    del df[df.columns[0]]

    df = df.dropna(axis=0)
    df = df.replace({"M": 1.0, "B": -1.0})

    data = np.array(df)
    train = data[:CANCER_TRAIN_SIZE, :]
    test = data[CANCER_TRAIN_SIZE:, :]

    return train, test[:, 0], test[:, 1:]


# ----------------------------------------------------------------------
# Tests and demos
# ----------------------------------------------------------------------


def test_cancer() -> None:
    """Train an RBF-kernel SVM on the breast cancer dataset and print accuracy."""
    train, test_labels, test_features = _load_cancer_data()

    kernel = Kernel("rbf", degree=5, coef0=1, gamma=0.5)
    initial_alphas = np.zeros(train.shape[0])
    model = SmoSVM(train, kernel, initial_alphas)

    model.fit()
    predictions = model.predict(test_features)

    accuracy = np.mean(predictions == test_labels)
    print(f"accuracy {accuracy}")


def demo() -> None:
    """Generate a 2×2 grid of decision-boundary plots for linear and RBF kernels."""
    # Suppress stdout during fitting to hide any incidental output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115

    fig, axes = plt.subplots(2, 2)

    _demo_linear(axes[0][0], cost=0.1)
    _demo_linear(axes[0][1], cost=500)
    _demo_rbf(axes[1][0], cost=0.1)
    _demo_rbf(axes[1][1], cost=500)

    sys.stdout = original_stdout
    print("plot ready")


def _make_labeled_dataset(
    features: np.ndarray, labels: np.ndarray, n_samples: int
) -> np.ndarray:
    """Combine labels and standardized features into a single array.

    Labels of 0 are converted to -1.

    Args:
        features: Raw feature matrix.
        labels: Integer label vector (0/1).
        n_samples: Number of samples (used for reshaping).

    Returns:
        Array of shape (n_samples, 1 + n_features).
    """
    labels = labels.copy()
    labels[labels == 0] = -1

    scaler = StandardScaler()
    features = scaler.fit_transform(features, labels)

    return np.hstack((labels.reshape(n_samples, 1), features))


def _demo_linear(ax: plt.Axes, cost: float) -> None:
    """Train and plot a linear-kernel SVM on linearly separable blobs."""
    n_samples = 500
    features, labels = make_blobs(
        n_samples=n_samples, centers=2, n_features=2, random_state=1
    )
    data = _make_labeled_dataset(features, labels, n_samples)

    kernel = Kernel("linear")
    model = SmoSVM(data, kernel, cost=cost, auto_norm=False)
    model.fit()

    _plot_decision_boundary(model, data, ax)


def _demo_rbf(ax: plt.Axes, cost: float) -> None:
    """Train and plot an RBF-kernel SVM on concentric circles."""
    n_samples = 500
    features, labels = make_circles(
        n_samples=n_samples, noise=0.1, factor=0.1, random_state=1
    )
    data = _make_labeled_dataset(features, labels, n_samples)

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
    """Plot data points, support vectors, and the decision boundary contour.

    Args:
        model: Trained SmoSVM model.
        data: Full dataset (labels in column 0).
        ax: Matplotlib axes to draw on.
        resolution: Grid resolution for the contour plot.
    """
    feature_x = data[:, 1]
    feature_y = data[:, 2]
    labels = data[:, 0]

    x_range = np.linspace(feature_x.min(), feature_x.max(), resolution)
    y_range = np.linspace(feature_y.min(), feature_y.max(), resolution)

    # Build a grid of evaluation points
    grid_points = np.array(
        [(a, b) for a in x_range for b in y_range]
    ).reshape(resolution * resolution, 2)

    # Compute raw decision values over the grid
    decision_values = model.predict(grid_points, classify=False)
    decision_grid = decision_values.reshape((resolution, resolution))

    # Plot decision boundary and margins
    ax.contour(
        x_range, y_range, np.asmatrix(decision_grid).T, levels=(-1, 0, 1)
    )

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


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    test_cancer()
    demo()
    plt.show()
