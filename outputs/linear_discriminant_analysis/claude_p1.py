"""Linear Discriminant Analysis (LDA) classifier with interactive input."""

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List, Optional, Type


def generate_gaussian_samples(mean: float, std_dev: float, count: int) -> List[float]:
    """Generate a list of Gaussian-distributed random values.

    Args:
        mean: The mean of the distribution.
        std_dev: The standard deviation of the distribution.
        count: The number of samples to generate.

    Returns:
        A list of Gaussian-distributed float values.
    """
    seed(1)
    return [gauss(mean, std_dev) for _ in range(count)]


def generate_class_labels(num_classes: int, counts: List[int]) -> List[int]:
    """Generate a flat list of class labels based on per-class instance counts.

    Args:
        num_classes: The number of distinct classes.
        counts: A list where counts[i] is the number of instances for class i.

    Returns:
        A list of integer class labels (e.g., [0, 0, 1, 1, 1]).
    """
    return [
        class_idx
        for class_idx in range(num_classes)
        for _ in range(counts[class_idx])
    ]


def calculate_mean(values: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        values: A non-empty list of numeric values.

    Returns:
        The arithmetic mean.
    """
    return sum(values) / len(values)


def calculate_prior_probability(class_count: int, total_count: int) -> float:
    """Calculate the prior probability of a class.

    Args:
        class_count: Number of instances belonging to the class.
        total_count: Total number of instances across all classes.

    Returns:
        The prior probability as a float.
    """
    return class_count / total_count


def calculate_pooled_variance(
    class_data: List[List[float]], class_means: List[float], total_count: int
) -> float:
    """Calculate the pooled within-class variance across all classes.

    Uses the unbiased estimator with (N - K) degrees of freedom,
    where N is the total sample count and K is the number of classes.

    Args:
        class_data: A list of lists, where each sublist contains samples for a class.
        class_means: The mean value for each class.
        total_count: The total number of samples across all classes.

    Returns:
        The pooled variance estimate.
    """
    num_classes = len(class_means)
    sum_of_squared_diffs = sum(
        (value - class_means[class_idx]) ** 2
        for class_idx, samples in enumerate(class_data)
        for value in samples
    )
    degrees_of_freedom = total_count - num_classes
    return sum_of_squared_diffs / degrees_of_freedom


def predict_classes(
    class_data: List[List[float]],
    class_means: List[float],
    variance: float,
    prior_probabilities: List[float],
) -> List[int]:
    """Predict class labels using the LDA discriminant function.

    For each data point, the discriminant score for each class is computed as:
        score_k = x * (mean_k / variance) - (mean_k^2 / (2 * variance)) + log(P(k))
    The class with the highest score is selected as the prediction.

    Args:
        class_data: A list of lists containing samples grouped by class.
        class_means: The mean of each class.
        variance: The pooled within-class variance.
        prior_probabilities: The prior probability of each class.

    Returns:
        A flat list of predicted class labels for all data points.
    """
    num_classes = len(class_means)
    predictions = []

    for samples in class_data:
        for value in samples:
            scores = [
                value * (class_means[k] / variance)
                - (class_means[k] ** 2) / (2 * variance)
                + log(prior_probabilities[k])
                for k in range(num_classes)
            ]
            predicted_class = scores.index(max(scores))
            predictions.append(predicted_class)

    return predictions


def calculate_accuracy(actual: List[int], predicted: List[int]) -> float:
    """Calculate classification accuracy as a percentage.

    Args:
        actual: The true class labels.
        predicted: The predicted class labels.

    Returns:
        Accuracy as a percentage (0–100).
    """
    correct = sum(a == p for a, p in zip(actual, predicted))
    return (correct / len(actual)) * 100


def validated_input(
    target_type: Type,
    prompt: str,
    error_message: str,
    condition: Callable[[Any], bool] = lambda x: True,
    default: Optional[str] = None,
) -> Any:
    """Prompt the user for input with type conversion, validation, and optional default.

    Args:
        target_type: The type to convert the raw input to (e.g., int, float).
        prompt: The message displayed to the user.
        error_message: The message shown when the condition check fails.
        condition: A callable that returns True if the converted value is valid.
        default: A default string value used when the user provides empty input.

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
            print("Invalid input. Please try again.")


def clear_console() -> None:
    """Clear the terminal screen in a cross-platform manner."""
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)


def main() -> None:
    """Run the interactive Linear Discriminant Analysis workflow."""
    while True:
        print("Linear Discriminant Analysis")
        print("-" * 42)

        # --- Gather user inputs ---
        num_classes = validated_input(
            int,
            "Enter number of classes: ",
            "Must be a positive integer.",
            lambda x: x > 0,
        )

        std_dev = validated_input(
            float,
            "Enter std dev (default 1.0): ",
            "Must be non-negative.",
            lambda x: x >= 0,
            default="1.0",
        )

        counts = [
            validated_input(
                int,
                f"Instances for class {i + 1}: ",
                "Must be a positive integer.",
                lambda x: x > 0,
            )
            for i in range(num_classes)
        ]

        means = [
            validated_input(
                float,
                f"Mean for class {i + 1}: ",
                "Invalid value.",
            )
            for i in range(num_classes)
        ]

        print(f"Standard deviation: {std_dev}")

        # --- Generate dataset ---
        class_data = [
            generate_gaussian_samples(means[i], std_dev, counts[i])
            for i in range(num_classes)
        ]
        print(f"Data: {class_data}")

        labels = generate_class_labels(num_classes, counts)
        print(f"Labels: {labels}")

        # --- Compute statistics ---
        actual_means = [calculate_mean(samples) for samples in class_data]
        for i, mean in enumerate(actual_means):
            print(f"Actual mean class {i + 1}: {mean}")

        total_samples = sum(counts)
        prior_probs = [
            calculate_prior_probability(count, total_samples) for count in counts
        ]
        for i, prob in enumerate(prior_probs):
            print(f"Prior probability class {i + 1}: {prob}")

        variance = calculate_pooled_variance(class_data, actual_means, total_samples)
        print(f"Pooled variance: {variance}")

        # --- Predict and evaluate ---
        predictions = predict_classes(class_data, actual_means, variance, prior_probs)
        acc = calculate_accuracy(labels, predictions)
        print(f"Accuracy: {acc:.2f}%")

        # --- Restart or quit ---
        user_choice = input("Press any key to restart or 'q' to quit: ").strip().lower()
        if user_choice == "q":
            print("Goodbye!")
            break

        clear_console()


if __name__ == "__main__":
    main()
