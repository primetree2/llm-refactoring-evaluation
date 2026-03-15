from __future__ import annotations

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List, Optional, Sequence, Type


def gaussian_distribution(m: float, s: float, n: int) -> List[float]:
    """Generate n samples from a Gaussian distribution with given mean and std.

    Note: Seeds the random number generator on every call to preserve
    original deterministic behavior.
    """
    seed(1)
    return [gauss(m, s) for _ in range(n)]


def y_generator(c: int, counts: Sequence[int]) -> List[int]:
    """Generate flat list of class labels from per-class counts.

    Args:
        c: Number of classes.
        counts: Number of instances per class.

    Returns:
        List of class labels in order.
    """
    result: List[int] = []
    for i in range(c):
        result.extend([i] * counts[i])
    return result


def calculate_mean(n: int, items: Sequence[float]) -> float:
    """Calculate arithmetic mean of items.

    Args:
        n: Number of items (preserved for original signature).
        items: Values to average.

    Returns:
        The mean value.
    """
    total = 0.0
    for x in items:
        total += x
    return total / n


def calculate_probabilities(n: int, total: int) -> float:
    """Calculate prior probability for a class.

    Args:
        n: Count for this class.
        total: Total samples across all classes.

    Returns:
        Prior probability.
    """
    return n / total


def calculate_variance(
    items: Sequence[Sequence[float]], means: Sequence[float], total_count: int
) -> float:
    """Calculate pooled within-class variance.

    Args:
        items: Samples grouped by class.
        means: Mean for each class.
        total_count: Total number of samples.

    Returns:
        Pooled variance using (N - K) denominator.
    """
    sum_of_squares = 0.0
    num_classes = len(means)

    for i, class_samples in enumerate(items):
        class_mean = means[i]
        for val in class_samples:
            diff = val - class_mean
            sum_of_squares += diff * diff

    return sum_of_squares / (total_count - num_classes)


def predict_y_values(
    x_items: Sequence[Sequence[float]],
    means: Sequence[float],
    variance: float,
    probabilities: Sequence[float],
) -> List[int]:
    """Predict class for each sample using LDA discriminant scores.

    Args:
        x_items: Samples grouped by class.
        means: Mean per class.
        variance: Pooled variance.
        probabilities: Prior probability per class.

    Returns:
        Predicted class indices in flattened order.
    """
    predictions: List[int] = []
    num_classes = len(x_items)

    for class_samples in x_items:
        for x in class_samples:
            scores: List[float] = []
            for k in range(num_classes):
                score = (
                    x * (means[k] / variance)
                    - (means[k] ** 2) / (2 * variance)
                    + log(probabilities[k])
                )
                scores.append(score)

            # Find index of maximum score
            best_idx = 0
            best_val = scores[0]
            for idx, val in enumerate(scores):
                if val > best_val:
                    best_val = val
                    best_idx = idx

            predictions.append(best_idx)

    return predictions


def accuracy(actual: Sequence[int], pred: Sequence[int]) -> float:
    """Calculate classification accuracy as percentage.

    Args:
        actual: True labels.
        pred: Predicted labels.

    Returns:
        Accuracy in percent.
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct += 1
    return (correct / len(actual)) * 100


def valid_input(
    tp: Type[Any],
    msg: str,
    err: str,
    cond: Callable[[Any], bool] = lambda x: True,
    default: Optional[str] = None,
) -> Any:
    """Get validated input from user with type conversion.

    Preserves original error handling and messaging behavior.
    """
    while True:
        raw = input(msg).strip()

        if not raw and default is not None:
            raw = default

        try:
            val = tp(raw)
            if cond(val):
                return val
            print(f"{val}: {err}")
        except Exception:
            print("bad input")


def main() -> None:
    """Main interactive loop for Linear Discriminant Analysis demo."""
    while True:
        print("Linear Discriminant Analysis")
        print("------------------------------------------")

        n_classes = valid_input(
            int,
            "Enter number of classes: ",
            "must be positive",
            lambda x: x > 0,
        )

        std = valid_input(
            float,
            "Enter std dev (default 1.0): ",
            "must not be negative",
            lambda x: x >= 0,
            "1.0",
        )

        counts: List[int] = []
        for i in range(n_classes):
            c = valid_input(
                int,
                f"instances for class {i + 1}: ",
                "must be positive",
                lambda x: x > 0,
            )
            counts.append(c)

        means: List[float] = []
        for i in range(n_classes):
            m = valid_input(
                float,
                f"mean for class {i + 1}: ",
                "invalid",
            )
            means.append(m)

        print("std dev:", std)

        # Generate synthetic data per class
        x: List[List[float]] = []
        for i in range(n_classes):
            g = gaussian_distribution(means[i], std, counts[i])
            x.append(g)

        print("data:", x)

        y = y_generator(n_classes, counts)
        print("labels:", y)

        # Compute actual means from generated data
        actual_means: List[float] = []
        for i in range(n_classes):
            m = calculate_mean(counts[i], x[i])
            actual_means.append(m)
            print("actual mean class", i + 1, ":", m)

        # Compute prior probabilities
        total = sum(counts)
        probs: List[float] = []
        for i in range(n_classes):
            p = calculate_probabilities(counts[i], total)
            probs.append(p)
            print("prob class", i + 1, ":", p)

        var = calculate_variance(x, actual_means, total)
        print("variance:", var)

        pred = predict_y_values(x, actual_means, var, probs)
        acc = accuracy(y, pred)
        print("accuracy:", acc)

        inp = input("press key to restart or q to quit: ").strip().lower()
        if inp == "q":
            print("bye")
            break

        # Clear screen
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")


if __name__ == "__main__":
    main()
