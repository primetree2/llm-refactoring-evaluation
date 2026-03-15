"""
Interactive Linear Discriminant Analysis (LDA) demo.

This module generates Gaussian-distributed samples for multiple classes,
computes class statistics, predicts class labels using a simple LDA rule,
and reports classification accuracy.
"""

import os
from math import log
from random import gauss, seed
from typing import Any, Callable


def gaussian_distribution(mean: float, std_dev: float, count: int) -> list[float]:
    """
    Generate Gaussian-distributed values.

    Note:
        The random generator is seeded on every call to preserve the original
        behavior of the program.

    Args:
        mean: Mean of the Gaussian distribution.
        std_dev: Standard deviation of the Gaussian distribution.
        count: Number of values to generate.

    Returns:
        A list of generated values.
    """
    seed(1)
    return [gauss(mean, std_dev) for _ in range(count)]


def y_generator(class_count: int, counts: list[int]) -> list[int]:
    """
    Generate class labels for the dataset.

    Args:
        class_count: Number of classes.
        counts: Number of samples for each class.

    Returns:
        A flat list of integer class labels.
    """
    labels: list[int] = []
    for class_index in range(class_count):
        labels.extend([class_index] * counts[class_index])
    return labels


def calculate_mean(count: int, items: list[float]) -> float:
    """
    Calculate the arithmetic mean.

    Args:
        count: Number of items.
        items: Values whose mean should be calculated.

    Returns:
        The mean of the provided values.
    """
    return sum(items) / count


def calculate_probabilities(count: int, total: int) -> float:
    """
    Calculate the prior probability of a class.

    Args:
        count: Number of instances in the class.
        total: Total number of instances.

    Returns:
        The class probability.
    """
    return count / total


def calculate_variance(
    items: list[list[float]],
    means: list[float],
    total_count: int,
) -> float:
    """
    Calculate the pooled variance across all classes.

    Args:
        items: Nested list of class samples.
        means: Mean value for each class.
        total_count: Total number of samples across all classes.

    Returns:
        The pooled variance estimate.
    """
    squared_differences = sum(
        (value - means[class_index]) ** 2
        for class_index, class_items in enumerate(items)
        for value in class_items
    )
    class_count = len(means)
    return (1 / (total_count - class_count)) * squared_differences


def predict_y_values(
    x_items: list[list[float]],
    means: list[float],
    variance: float,
    probabilities: list[float],
) -> list[int]:
    """
    Predict class labels using a linear discriminant score.

    Args:
        x_items: Nested list of samples grouped by class.
        means: Mean value for each class.
        variance: Shared variance across classes.
        probabilities: Prior probability for each class.

    Returns:
        A list of predicted class indices.
    """
    predictions: list[int] = []

    for class_items in x_items:
        for value in class_items:
            scores = [
                value * (means[class_index] / variance)
                - ((means[class_index] ** 2) / (2 * variance))
                + log(probabilities[class_index])
                for class_index in range(len(x_items))
            ]

            best_index = 0
            best_score = scores[0]
            for index, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_index = index

            predictions.append(best_index)

    return predictions


def accuracy(actual: list[int], pred: list[int]) -> float:
    """
    Calculate classification accuracy as a percentage.

    Args:
        actual: Ground-truth labels.
        pred: Predicted labels.

    Returns:
        Accuracy percentage.
    """
    correct = sum(1 for actual_value, pred_value in zip(actual, pred) if actual_value == pred_value)
    return (correct / len(actual)) * 100


def valid_input(
    tp: Callable[[str], Any],
    msg: str,
    err: str,
    cond: Callable[[Any], bool] = lambda x: True,
    default: str | None = None,
) -> Any:
    """
    Prompt for input until a valid value is entered.

    Args:
        tp: Type or callable used to convert the raw input.
        msg: Prompt message displayed to the user.
        err: Error message shown when validation fails.
        cond: Validation function applied to the converted value.
        default: Default raw input value if the user enters nothing.

    Returns:
        The validated and converted input value.
    """
    while True:
        raw = input(msg).strip()

        if raw == "" and default is not None:
            raw = default

        try:
            value = tp(raw)
            if cond(value):
                return value
            print(f"{value}: {err}")
        except Exception:
            print("bad input")


def _clear_screen() -> None:
    """Clear the terminal screen based on the operating system."""
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    """Run the interactive Linear Discriminant Analysis workflow."""
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

        counts = [
            valid_input(
                int,
                f"instances for class {class_index + 1}: ",
                "must be positive",
                lambda x: x > 0,
            )
            for class_index in range(n_classes)
        ]

        means = [
            valid_input(
                float,
                f"mean for class {class_index + 1}: ",
                "invalid",
            )
            for class_index in range(n_classes)
        ]

        print("std dev:", std)

        # Generate synthetic dataset.
        x = [
            gaussian_distribution(means[class_index], std, counts[class_index])
            for class_index in range(n_classes)
        ]
        print("data:", x)

        y = y_generator(n_classes, counts)
        print("labels:", y)

        actual_means: list[float] = []
        for class_index in range(n_classes):
            mean_value = calculate_mean(counts[class_index], x[class_index])
            actual_means.append(mean_value)
            print("actual mean class", class_index + 1, ":", mean_value)

        total = sum(counts)
        probs: list[float] = []
        for class_index in range(n_classes):
            probability = calculate_probabilities(counts[class_index], total)
            probs.append(probability)
            print("prob class", class_index + 1, ":", probability)

        var = calculate_variance(x, actual_means, total)
        print("variance:", var)

        pred = predict_y_values(x, actual_means, var, probs)
        acc = accuracy(y, pred)
        print("accuracy:", acc)

        user_input = input("press key to restart or q to quit: ").strip().lower()
        if user_input == "q":
            print("bye")
            break

        _clear_screen()


if __name__ == "__main__":
    main()
