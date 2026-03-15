"""
Linear Discriminant Analysis (LDA) implementation.

This module provides a simple, from-scratch implementation of LDA for 1D data,
including synthetic data generation, parameter estimation, and prediction.
"""

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List, Optional


def generate_gaussian_samples(mean: float, std_dev: float, count: int) -> List[float]:
    """Generate a list of Gaussian-distributed values.

    Args:
        mean: The mean of the Gaussian distribution.
        std_dev: The standard deviation of the distribution.
        count: The number of values to generate.

    Returns:
        A list of generated float values.
    """
    seed(1)  # Kept to maintain the exact original functionality and reproducibility
    return [gauss(mean, std_dev) for _ in range(count)]


def generate_class_labels(num_classes: int, counts: List[int]) -> List[int]:
    """Generate a flattened list of class labels based on class instance counts.

    Args:
        num_classes: The total number of classes.
        counts: A list containing the number of instances for each class.

    Returns:
        A list of integer class labels.
    """
    return [
        class_idx
        for class_idx in range(num_classes)
        for _ in range(counts[class_idx])
    ]


def calculate_mean(count: int, items: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numeric values.

    Args:
        count: The number of items in the list.
        items: A list of numeric values.

    Returns:
        The computed mean.
    """
    return sum(items) / count


def calculate_prior_probability(class_count: int, total_count: int) -> float:
    """Calculate the prior probability of a specific class.

    Args:
        class_count: The number of items belonging to the class.
        total_count: The total number of items across all classes.

    Returns:
        The prior probability.
    """
    return class_count / total_count


def calculate_pooled_variance(
    class_data: List[List[float]], class_means: List[float], total_count: int
) -> float:
    """Calculate the pooled within-class variance.

    Args:
        class_data: A list containing lists of samples for each class.
        class_means: A list of means for each class.
        total_count: The total number of samples across all classes.

    Returns:
        The pooled variance.
    """
    num_classes = len(class_means)
    sum_squared_diffs = sum(
        (val - class_means[class_idx]) ** 2
        for class_idx, samples in enumerate(class_data)
        for val in samples
    )

    return sum_squared_diffs / (total_count - num_classes)


def predict_classes(
    class_data: List[List[float]],
    class_means: List[float],
    variance: float,
    prior_probs: List[float],
) -> List[int]:
    """Predict class labels using the LDA discriminant function.

    Args:
        class_data: A list containing lists of samples for each class.
        class_means: A list of means for each class.
        variance: The pooled variance of the dataset.
        prior_probs: A list of prior probabilities for each class.

    Returns:
        A flat list of predicted class indices.
    """
    predictions = []
    num_classes = len(class_means)

    for samples in class_data:
        for val in samples:
            # Calculate discriminant scores for each class
            scores = [
                val * (class_means[k] / variance)
                - ((class_means[k] ** 2) / (2 * variance))
                + log(prior_probs[k])
                for k in range(num_classes)
            ]
            # Predict the class with the highest score
            predicted_class = scores.index(max(scores))
            predictions.append(predicted_class)

    return predictions


def calculate_accuracy(actual: List[int], predicted: List[int]) -> float:
    """Calculate the percentage of correct predictions.

    Args:
        actual: The true class labels.
        predicted: The predicted class labels.

    Returns:
        The accuracy as a percentage.
    """
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return (correct / len(actual)) * 100


def get_validated_input(
    target_type: type,
    prompt: str,
    error_msg: str,
    condition: Callable[[Any], bool] = lambda x: True,
    default: Optional[str] = None,
) -> Any:
    """Prompt the user for input until a valid response is provided.

    Args:
        target_type: The expected data type (e.g., int, float).
        prompt: The input prompt displayed to the user.
        error_msg: The error message to display on validation failure.
        condition: A function that takes the casted value and returns a boolean.
        default: An optional default string to use if the user enters nothing.

    Returns:
        The validated and casted user input.
    """
    while True:
        raw = input(prompt).strip()

        if raw == "" and default is not None:
            raw = default

        try:
            val = target_type(raw)
            if condition(val):
                return val
            print(f"{val}: {error_msg}")
        except ValueError:
            print("Invalid input format. Please try again.")


def clear_console() -> None:
    """Clear the terminal console."""
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    """Main execution loop for the LDA script."""
    while True:
        print("Linear Discriminant Analysis")
        print("-" * 42)

        n_classes = get_validated_input(
            int,
            "Enter number of classes: ",
            "must be positive",
            lambda x: x > 0,
        )

        std_dev = get_validated_input(
            float,
            "Enter std dev (default 1.0): ",
            "must not be negative",
            lambda x: x >= 0,
            default="1.0",
        )

        counts = [
            get_validated_input(
                int,
                f"Instances for class {i + 1}: ",
                "must be positive",
                lambda x: x > 0,
            )
            for i in range(n_classes)
        ]

        means = [
            get_validated_input(
                float,
                f"Mean for class {i + 1}: ",
                "invalid",
            )
            for i in range(n_classes)
        ]

        print(f"Standard deviation: {std_dev}")

        # Generate dataset
        x_data = [
            generate_gaussian_samples(means[i], std_dev, counts[i])
            for i in range(n_classes)
        ]
        print(f"Data: {x_data}")

        y_actual = generate_class_labels(n_classes, counts)
        print(f"Labels: {y_actual}")

        # Calculate dataset statistics
        actual_means = []
        for i in range(n_classes):
            mean_val = calculate_mean(counts[i], x_data[i])
            actual_means.append(mean_val)
            print(f"Actual mean class {i + 1}: {mean_val}")

        total_samples = sum(counts)
        prior_probs = []
        for i in range(n_classes):
            prob = calculate_prior_probability(counts[i], total_samples)
            prior_probs.append(prob)
            print(f"Prob class {i + 1}: {prob}")

        variance = calculate_pooled_variance(x_data, actual_means, total_samples)
        print(f"Variance: {variance}")

        # Predict and evaluate
        predictions = predict_classes(x_data, actual_means, variance, prior_probs)
        accuracy = calculate_accuracy(y_actual, predictions)
        print(f"Accuracy: {accuracy:.2f}%")

        # Restart or exit prompt
        user_choice = input("Press enter to restart or 'q' to quit: ").strip().lower()
        if user_choice == "q":
            print("Goodbye")
            break

        clear_console()


if __name__ == "__main__":
    main()
