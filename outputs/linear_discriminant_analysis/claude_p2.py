"""
Linear Discriminant Analysis (LDA) implementation.

This module provides a simple LDA classifier that generates Gaussian-distributed
data, computes class statistics, and predicts class labels using discriminant
functions.
"""

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List, Optional, Type


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------


def generate_gaussian_samples(
    mean: float, std_dev: float, num_samples: int
) -> List[float]:
    """
    Generate a list of values drawn from a Gaussian distribution.

    Args:
        mean: Mean of the Gaussian distribution.
        std_dev: Standard deviation of the Gaussian distribution.
        num_samples: Number of samples to generate.

    Returns:
        A list of Gaussian-distributed float values.
    """
    seed(1)
    return [gauss(mean, std_dev) for _ in range(num_samples)]


def generate_class_labels(num_classes: int, counts: List[int]) -> List[int]:
    """
    Generate a flat list of class labels based on per-class instance counts.

    Args:
        num_classes: Number of distinct classes.
        counts: A list where counts[i] is the number of instances for class i.

    Returns:
        A list of integer class labels (0-indexed).
    """
    return [
        class_idx
        for class_idx in range(num_classes)
        for _ in range(counts[class_idx])
    ]


# ---------------------------------------------------------------------------
# Statistical Calculations
# ---------------------------------------------------------------------------


def calculate_mean(values: List[float]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    Args:
        values: A non-empty list of numeric values.

    Returns:
        The arithmetic mean.
    """
    return sum(values) / len(values)


def calculate_prior_probability(class_count: int, total_count: int) -> float:
    """
    Calculate the prior probability of a class.

    Args:
        class_count: Number of instances belonging to the class.
        total_count: Total number of instances across all classes.

    Returns:
        The prior probability as a float.
    """
    return class_count / total_count


def calculate_pooled_variance(
    class_data: List[List[float]],
    class_means: List[float],
    total_count: int,
) -> float:
    """
    Calculate the pooled within-class variance.

    Uses the unbiased estimator with (N - K) in the denominator, where
    N is the total sample count and K is the number of classes.

    Args:
        class_data: A list of lists, where each inner list contains the
            feature values for one class.
        class_means: The mean feature value for each class.
        total_count: Total number of instances across all classes.

    Returns:
        The pooled variance estimate.
    """
    num_classes = len(class_means)
    sum_of_squared_diffs = sum(
        (value - class_means[class_idx]) ** 2
        for class_idx, values in enumerate(class_data)
        for value in values
    )
    return sum_of_squared_diffs / (total_count - num_classes)


# ---------------------------------------------------------------------------
# Prediction & Evaluation
# ---------------------------------------------------------------------------


def predict_class_labels(
    class_data: List[List[float]],
    class_means: List[float],
    variance: float,
    prior_probabilities: List[float],
) -> List[int]:
    """
    Predict class labels for every observation using LDA discriminant functions.

    For each observation x, the discriminant score for class k is:
        score_k = x * (mean_k / variance)
                  - (mean_k^2 / (2 * variance))
                  + ln(prior_k)

    The predicted class is the one with the highest score.

    Args:
        class_data: Feature values grouped by class.
        class_means: Mean feature value for each class.
        variance: Pooled within-class variance.
        prior_probabilities: Prior probability for each class.

    Returns:
        A list of predicted class indices.
    """
    num_classes = len(class_means)
    predictions: List[int] = []

    for class_values in class_data:
        for value in class_values:
            # Compute discriminant score for every candidate class
            scores = [
                value * (class_means[k] / variance)
                - (class_means[k] ** 2 / (2 * variance))
                + log(prior_probabilities[k])
                for k in range(num_classes)
            ]
            # Assign the class with the highest discriminant score
            predictions.append(scores.index(max(scores)))

    return predictions


def calculate_accuracy(actual: List[int], predicted: List[int]) -> float:
    """
    Calculate classification accuracy as a percentage.

    Args:
        actual: Ground-truth class labels.
        predicted: Predicted class labels.

    Returns:
        Accuracy in the range [0, 100].
    """
    correct = sum(a == p for a, p in zip(actual, predicted))
    return (correct / len(actual)) * 100


# ---------------------------------------------------------------------------
# User Input Helpers
# ---------------------------------------------------------------------------


def validated_input(
    target_type: Type,
    prompt: str,
    error_message: str,
    condition: Callable[[Any], bool] = lambda x: True,
    default: Optional[str] = None,
) -> Any:
    """
    Repeatedly prompt the user until valid input is received.

    Args:
        target_type: The type to which the raw input string is converted.
        prompt: The message displayed to the user.
        error_message: Message shown when the condition check fails.
        condition: A callable returning True if the converted value is valid.
        default: A fallback string used when the user enters nothing.

    Returns:
        The validated and type-converted input value.
    """
    while True:
        raw = input(prompt).strip()

        if raw == "" and default is not None:
            raw = default

        try:
            value = target_type(raw)
            if condition(value):
                return value
            print(f"{value}: {error_message}")
        except (ValueError, TypeError):
            print("bad input")


def clear_terminal() -> None:
    """Clear the terminal screen in a cross-platform manner."""
    os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the interactive Linear Discriminant Analysis demo."""
    while True:
        print("Linear Discriminant Analysis")
        print("------------------------------------------")

        # Collect configuration from the user
        num_classes = validated_input(
            int, "Enter number of classes: ", "must be positive", lambda x: x > 0
        )

        std_dev = validated_input(
            float,
            "Enter std dev (default 1.0): ",
            "must not be negative",
            lambda x: x >= 0,
            "1.0",
        )

        counts: List[int] = [
            validated_input(
                int,
                f"instances for class {i + 1}: ",
                "must be positive",
                lambda x: x > 0,
            )
            for i in range(num_classes)
        ]

        means: List[float] = [
            validated_input(float, f"mean for class {i + 1}: ", "invalid")
            for i in range(num_classes)
        ]

        print(f"std dev: {std_dev}")

        # Generate synthetic Gaussian data for each class
        class_data = [
            generate_gaussian_samples(means[i], std_dev, counts[i])
            for i in range(num_classes)
        ]
        print(f"data: {class_data}")

        # Generate ground-truth labels
        labels = generate_class_labels(num_classes, counts)
        print(f"labels: {labels}")

        # Compute per-class statistics
        total_count = sum(counts)
        actual_means = []
        prior_probs = []

        for i in range(num_classes):
            mean_i = calculate_mean(class_data[i])
            actual_means.append(mean_i)
            print(f"actual mean class {i + 1} : {mean_i}")

            prob_i = calculate_prior_probability(counts[i], total_count)
            prior_probs.append(prob_i)
            print(f"prob class {i + 1} : {prob_i}")

        variance = calculate_pooled_variance(class_data, actual_means, total_count)
        print(f"variance: {variance}")

        # Predict and evaluate
        predictions = predict_class_labels(
            class_data, actual_means, variance, prior_probs
        )
        acc = calculate_accuracy(labels, predictions)
        print(f"accuracy: {acc}")

        # Prompt to continue or quit
        if input("press key to restart or q to quit: ").strip().lower() == "q":
            print("bye")
            break

        clear_terminal()


if __name__ == "__main__":
    main()
