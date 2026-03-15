"""
Linear Discriminant Analysis (LDA) demo.

Generates synthetic Gaussian data for multiple classes, computes class
statistics (means, priors, pooled variance), predicts class labels using
discriminant functions, and reports accuracy.
"""

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List


def gaussian_distribution(mean: float, std: float, n: int) -> List[float]:
    """
    Generate n samples from a Gaussian distribution with given mean and std.

    Note:
        The random seed is reset on every call to preserve original behavior.
    """
    seed(1)
    return [gauss(mean, std) for _ in range(n)]


def y_generator(num_classes: int, counts: List[int]) -> List[int]:
    """Generate a flat list of class labels based on per-class counts."""
    labels: List[int] = []
    for class_idx in range(num_classes):
        labels.extend([class_idx] * counts[class_idx])
    return labels


def calculate_mean(count: int, values: List[float]) -> float:
    """Calculate the arithmetic mean of a list of values."""
    return sum(values) / count


def calculate_probabilities(count: int, total: int) -> float:
    """Calculate the prior probability of a class."""
    return count / total


def calculate_variance(
    class_data: List[List[float]], means: List[float], total_count: int
) -> float:
    """
    Calculate the pooled within-class variance.

    Uses the unbiased estimator with (N - K) in the denominator.
    """
    sum_sq_diff = sum(
        (x - means[i]) ** 2
        for i, class_samples in enumerate(class_data)
        for x in class_samples
    )
    return sum_sq_diff / (total_count - len(means))


def predict_y_values(
    x_items: List[List[float]],
    means: List[float],
    variance: float,
    probabilities: List[float],
) -> List[int]:
    """
    Predict class labels using linear discriminant analysis scores.

    For each observation x the score for class k is:
        x * (mean_k / variance) - (mean_k² / (2 * variance)) + ln(prior_k)
    """
    predictions: List[int] = []
    num_classes = len(means)

    for class_samples in x_items:
        for x in class_samples:
            scores = [
                x * (means[k] / variance)
                - (means[k] ** 2 / (2 * variance))
                + log(probabilities[k])
                for k in range(num_classes)
            ]
            predictions.append(scores.index(max(scores)))

    return predictions


def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Calculate classification accuracy as a percentage."""
    correct = sum(a == p for a, p in zip(actual, predicted))
    return (correct / len(actual)) * 100


def valid_input(
    target_type: Callable[[str], Any],
    prompt: str,
    error_msg: str,
    condition: Callable[[Any], bool] = lambda x: True,
    default: str | None = None,
) -> Any:
    """
    Prompt the user for input until a valid value is provided.

    Args:
        target_type: Type or callable to convert the input string.
        prompt: Message shown to the user.
        error_msg: Message shown when validation fails.
        condition: Function that returns True if the value is acceptable.
        default: Optional default value used when user enters an empty line.
    """
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            raw = default

        try:
            value = target_type(raw)
            if condition(value):
                return value
            print(f"{value}: {error_msg}")
        except Exception:
            print("bad input")


def clear_screen() -> None:
    """Clear the terminal screen (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    """Run the interactive LDA demonstration."""
    while True:
        print("Linear Discriminant Analysis")
        print("------------------------------------------")

        num_classes = valid_input(
            int,
            "Enter number of classes: ",
            "must be positive",
            lambda x: x > 0,
        )

        std_dev = valid_input(
            float,
            "Enter std dev (default 1.0): ",
            "must not be negative",
            lambda x: x >= 0,
            "1.0",
        )

        counts = [
            valid_input(
                int,
                f"instances for class {i + 1}: ",
                "must be positive",
                lambda x: x > 0,
            )
            for i in range(num_classes)
        ]

        means = [
            valid_input(
                float,
                f"mean for class {i + 1}: ",
                "invalid",
            )
            for i in range(num_classes)
        ]

        print("std dev:", std_dev)

        # Generate synthetic dataset
        dataset = [
            gaussian_distribution(means[i], std_dev, counts[i])
            for i in range(num_classes)
        ]
        print("data:", dataset)

        labels = y_generator(num_classes, counts)
        print("labels:", labels)

        # Compute class statistics
        total_samples = sum(counts)
        actual_means: List[float] = []
        priors: List[float] = []

        for i in range(num_classes):
            mean_val = calculate_mean(counts[i], dataset[i])
            actual_means.append(mean_val)
            print("actual mean class", i + 1, ":", mean_val)

            prob = calculate_probabilities(counts[i], total_samples)
            priors.append(prob)
            print("prob class", i + 1, ":", prob)

        variance = calculate_variance(dataset, actual_means, total_samples)
        print("variance:", variance)

        predictions = predict_y_values(dataset, actual_means, variance, priors)
        acc = accuracy(labels, predictions)
        print("accuracy:", acc)

        choice = input("press key to restart or q to quit: ").strip().lower()
        if choice == "q":
            print("bye")
            break

        clear_screen()


if __name__ == "__main__":
    main()
