from __future__ import annotations

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List, Optional, Sequence, Type


def gaussian_distribution(m: float, s: float, n: int) -> List[float]:
    """Generate `n` Gaussian-distributed samples with mean `m` and std dev `s`.

    Note:
        This function intentionally seeds the RNG on every call to preserve the
        original script's deterministic behavior.
    """
    seed(1)
    return [gauss(m, s) for _ in range(n)]


def y_generator(c: int, counts: Sequence[int]) -> List[int]:
    """Generate class labels for `c` classes given per-class instance counts.

    Args:
        c: Number of classes.
        counts: A sequence where `counts[i]` is the number of instances in class `i`.

    Returns:
        A flat list of integer labels of length `sum(counts)`.
    """
    labels: List[int] = []
    for class_idx in range(c):
        labels.extend([class_idx] * counts[class_idx])
    return labels


def calculate_mean(n: int, items: Sequence[float]) -> float:
    """Calculate the mean of `items`, given `n` items (kept for API compatibility)."""
    total = 0.0
    for x in items:
        total += x
    return total / n


def calculate_probabilities(n: int, total: int) -> float:
    """Calculate the prior probability of a class given its count and total count."""
    return n / total


def calculate_variance(
    items: Sequence[Sequence[float]], means: Sequence[float], total_count: int
) -> float:
    """Compute pooled within-class variance using (N - K) degrees of freedom.

    Args:
        items: A sequence of classes, where each class is a sequence of samples.
        means: Mean value for each class.
        total_count: Total number of samples across all classes (N).

    Returns:
        Pooled variance estimate.
    """
    num_classes = len(means)
    sum_squared_diffs = 0.0

    for class_idx, samples in enumerate(items):
        class_mean = means[class_idx]
        for value in samples:
            diff = value - class_mean
            sum_squared_diffs += diff * diff

    return (1 / (total_count - num_classes)) * sum_squared_diffs


def predict_y_values(
    x_items: Sequence[Sequence[float]],
    means: Sequence[float],
    variance: float,
    probabilities: Sequence[float],
) -> List[int]:
    """Predict class labels for each sample using a 1D LDA discriminant function.

    Args:
        x_items: Samples grouped by their true class (list of lists).
        means: Estimated mean for each class.
        variance: Pooled within-class variance.
        probabilities: Prior probability for each class.

    Returns:
        A flat list of predicted class indices, aligned with the flattened sample order.
    """
    num_classes = len(x_items)
    predictions: List[int] = []

    for samples in x_items:
        for x in samples:
            scores: List[float] = []
            for k in range(num_classes):
                score = (
                    x * (means[k] / variance)
                    - ((means[k] ** 2) / (2 * variance))
                    + log(probabilities[k])
                )
                scores.append(score)

            # Choose the index of the maximum score; ties favor the first occurrence.
            best_idx = 0
            best_score = scores[0]
            for idx, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_idx = idx

            predictions.append(best_idx)

    return predictions


def accuracy(actual: Sequence[int], pred: Sequence[int]) -> float:
    """Compute prediction accuracy percentage."""
    correct = 0
    i = 0
    while i < len(actual):
        if actual[i] == pred[i]:
            correct += 1
        i += 1
    return (correct / len(actual)) * 100


def valid_input(
    tp: Type[Any],
    msg: str,
    err: str,
    cond: Callable[[Any], bool] = lambda x: True,
    default: Optional[str] = None,
) -> Any:
    """Read and validate user input.

    Args:
        tp: Type to cast the raw input to (e.g., int, float).
        msg: Input prompt.
        err: Error message printed when `cond` fails.
        cond: Validation predicate for the casted value.
        default: Optional default value (string) used when the user enters nothing.

    Returns:
        The validated value cast to `tp`.

    Notes:
        This intentionally preserves original behavior:
        - Broad exception handling with "bad input" message.
        - Default substitution prior to casting.
    """
    while True:
        raw = input(msg).strip()

        if raw == "" and default is not None:
            raw = default

        try:
            val = tp(raw)
            if cond(val):
                return val
            print(f"{val}: {err}")
        except Exception:
            print("bad input")


def main() -> None:
    """Interactive driver for generating data, fitting parameters, and evaluating LDA."""
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
            count = valid_input(
                int,
                f"instances for class {i + 1}: ",
                "must be positive",
                lambda x: x > 0,
            )
            counts.append(count)

        means: List[float] = []
        for i in range(n_classes):
            mean = valid_input(
                float,
                f"mean for class {i + 1}: ",
                "invalid",
            )
            means.append(mean)

        print("std dev:", std)

        # Generate dataset grouped by class
        x: List[List[float]] = []
        for i in range(n_classes):
            x.append(gaussian_distribution(means[i], std, counts[i]))

        print("data:", x)

        y = y_generator(n_classes, counts)
        print("labels:", y)

        # Estimate per-class means from generated data
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

        # Clear console (preserve original OS-specific behavior)
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
