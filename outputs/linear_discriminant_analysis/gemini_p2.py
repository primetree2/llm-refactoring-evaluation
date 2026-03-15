"""
Linear Discriminant Analysis (LDA) Implementation.

This module provides a simple implementation of LDA, including Gaussian data
generation, statistical calculations (mean, variance, prior probabilities),
and a prediction mechanism to classify generated data points.
"""

import os
from math import log
from random import gauss, seed
from typing import Any, Callable, List, Optional, Type


def generate_gaussian_samples(mean: float, std_dev: float, count: int) -> List[float]:
    """
    Generate a list of random values from a Gaussian distribution.

    Args:
        mean: The mean of the Gaussian distribution.
        std_dev: The standard deviation of the Gaussian distribution.
        count: The number of values to generate.

    Returns:
        A list of generated float values.
    """
    seed(1)
    return [gauss(mean, std_dev) for _ in range(count)]


def generate_class_labels(num_classes: int, instance_counts: List[int]) -> List[int]:
    """
    Generate a flat list of class labels for the dataset.

    Args:
        num_classes: The total number of classes.
        instance_counts: A list containing the number of instances for each class.

    Returns:
        A list of integer labels representing the class of each instance.
    """
    labels = []
    for class_index in range(num_classes):
        labels.extend([class_index] * instance_counts[class_index])
    return labels


def calculate_mean(items: List[float]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    Args:
        items: A list of numerical values.

    Returns:
        The mean of the values. Returns 0.0 if the list is empty.
    """
    if not items:
        return 0.0
    return sum(items) / len(items)


def calculate_prior_probability(class_count: int, total_count: int) -> float:
    """
    Calculate the prior probability of a class.

    Args:
        class_count: The number of instances in the class.
        total_count: The total number of instances across all classes.

    Returns:
        The prior probability as a float.
    """
    return class_count / total_count


def calculate_pooled_variance(
    data: List[List[float]], means: List[float], total_count: int
) -> float:
    """
    Calculate the pooled within-class variance of the dataset.

    Args:
        data: A list of lists, where each inner list contains instances for a class.
        means: A list of calculated means for each class.
        total_count: The total number of instances across all classes.

    Returns:
        The calculated pooled variance.
    """
    sum_squared_diffs = sum(
        (value - means[class_idx]) ** 2
        for class_idx, class_items in enumerate(data)
        for value in class_items
    )
    degrees_of_freedom = total_count - len(means)
    return sum_squared_diffs / degrees_of_freedom


def predict_class_labels(
    data: List[List[float]],
    means: List[float],
    variance: float,
    prior_probabilities: List[float],
) -> List[int]:
    """
    Predict the class labels for a dataset using discriminant functions.

    Args:
        data: The nested list of feature values for all instances.
        means: The list of class means.
        variance: The pooled variance.
        prior_probabilities: The list of prior probabilities for each class.

    Returns:
        A list of predicted class indices.
    """
    predictions = []
    num_classes = len(means)

    for class_items in data:
        for value in class_items:
            class_scores = []
            for k in range(num_classes):
                # Calculate the discriminant function score for class k
                score = (
                    value * (means[k] / variance)
                    - ((means[k] ** 2) / (2 * variance))
                    + log(prior_probabilities[k])
                )
                class_scores.append(score)
            
            # Select the class index with the maximum score
            best_class = class_scores.index(max(class_scores))
            predictions.append(best_class)

    return predictions


def calculate_accuracy(actual_labels: List[int], predicted_labels: List[int]) -> float:
    """
    Calculate the accuracy of predictions against actual labels.

    Args:
        actual_labels: The list of ground-truth labels.
        predicted_labels: The list of predicted labels.

    Returns:
        The accuracy as a percentage (0.0 to 100.0).
    """
    if not actual_labels:
        return 0.0

    correct_predictions = sum(
        1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted
    )
    return (correct_predictions / len(actual_labels)) * 100


def get_validated_input(
    target_type: Type,
    prompt: str,
    error_message: str,
    condition: Callable[[Any], bool] = lambda x: True,
    default_value: Optional[str] = None,
) -> Any:
    """
    Prompt the user for input and validate it against a type and a condition.

    Args:
        target_type: The expected type of the input (e.g., int, float).
        prompt: The message to display to the user.
        error_message: The message to show if the condition fails.
        condition: A function that takes the parsed value and returns True if valid.
        default_value: An optional default value used if the user enters nothing.

    Returns:
        The validated user input converted to the target type.
    """
    while True:
        raw_input = input(prompt).strip()

        if raw_input == "" and default_value is not None:
            raw_input = default_value

        try:
            parsed_value = target_type(raw_input)
            if condition(parsed_value):
                return parsed_value
            print(f"{parsed_value}: {error_message}")
        except ValueError:
            print("Invalid input format. Please try again.")


def clear_console() -> None:
    """Clear the console output."""
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    """
    Main function to run the interactive Linear Discriminant Analysis program.
    """
    while True:
        print("Linear Discriminant Analysis")
        print("-" * 42)

        num_classes = get_validated_input(
            int,
            "Enter number of classes: ",
            "must be positive",
            condition=lambda x: x > 0,
        )

        std_dev = get_validated_input(
            float,
            "Enter std dev (default 1.0): ",
            "must not be negative",
            condition=lambda x: x >= 0,
            default_value="1.0",
        )

        instance_counts = [
            get_validated_input(
                int,
                f"instances for class {i + 1}: ",
                "must be positive",
                condition=lambda x: x > 0,
            )
            for i in range(num_classes)
        ]

        class_means = [
            get_validated_input(
                float,
                f"mean for class {i + 1}: ",
                "invalid",
            )
            for i in range(num_classes)
        ]

        print(f"\nstd dev: {std_dev}")

        # Generate dataset and labels
        dataset = [
            generate_gaussian_samples(class_means[i], std_dev, instance_counts[i])
            for i in range(num_classes)
        ]
        print(f"data: {dataset}")

        actual_labels = generate_class_labels(num_classes, instance_counts)
        print(f"labels: {actual_labels}\n")

        # Calculate statistics
        total_instances = sum(instance_counts)
        calculated_means = []
        prior_probabilities = []

        for i in range(num_classes):
            mean_val = calculate_mean(dataset[i])
            calculated_means.append(mean_val)
            print(f"actual mean class {i + 1}: {mean_val}")

            prob_val = calculate_prior_probability(instance_counts[i], total_instances)
            prior_probabilities.append(prob_val)
            print(f"prob class {i + 1}: {prob_val}")

        variance = calculate_pooled_variance(dataset, calculated_means, total_instances)
        print(f"\nvariance: {variance}")

        # Predict and evaluate
        predicted_labels = predict_class_labels(
            dataset, calculated_means, variance, prior_probabilities
        )
        accuracy_score = calculate_accuracy(actual_labels, predicted_labels)

        print(f"accuracy: {accuracy_score:.2f}%\n")

        # Prompt for restart or exit
        user_choice = input("press key to restart or q to quit: ").strip().lower()
        if user_choice == "q":
            print("bye")
            break

        clear_console()


if __name__ == "__main__":
    main()
