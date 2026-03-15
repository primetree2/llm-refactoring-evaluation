

```python
"""Genetic algorithm to evolve a string toward a target string."""

import random
from typing import Optional

# --- Configuration Constants ---
POPULATION_SIZE = 200
NUM_SELECTED = 50
MUTATION_PROBABILITY = 0.4
MAX_CHILDREN_PER_PARENT = 10
KEEP_RATIO = 1 / 3
DEBUG_INTERVAL = 10


def evaluate(candidate: str, target: str) -> tuple[str, float]:
    """Evaluate the fitness of a candidate string against the target.

    Args:
        candidate: The string to evaluate.
        target: The target string to compare against.

    Returns:
        A tuple of (candidate, fitness_score) where fitness_score
        is the number of matching characters.
    """
    score = sum(c == t for c, t in zip(candidate, target))
    return (candidate, float(score))


def crossover(parent1: str, parent2: str) -> tuple[str, str]:
    """Perform single-point crossover between two parent strings.

    Args:
        parent1: The first parent string.
        parent2: The second parent string.

    Returns:
        A tuple of two child strings produced by crossover.
    """
    point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return (child1, child2)


def mutate(individual: str, genes: list[str]) -> str:
    """Randomly mutate a single position in the individual.

    A mutation occurs with probability defined by MUTATION_PROBABILITY.
    When it occurs, a random position is replaced with a random gene.

    Args:
        individual: The string to potentially mutate.
        genes: The list of valid characters to use for mutation.

    Returns:
        The (possibly mutated) string.
    """
    if random.random() < MUTATION_PROBABILITY:
        characters = list(individual)
        position = random.randint(0, len(individual) - 1)
        characters[position] = random.choice(genes)
        return "".join(characters)
    return individual


def select_and_breed(
    parent_entry: tuple[str, float],
    population_scores: list[tuple[str, float]],
    genes: list[str],
) -> list[str]:
    """Select a parent, breed with random partners, and produce children.

    The number of children produced is proportional to the parent's
    fitness score, capped at MAX_CHILDREN_PER_PARENT.

    Args:
        parent_entry: A (string, normalized_score) tuple for the primary parent.
        population_scores: The scored population to select partners from.
        genes: The list of valid characters for mutation.

    Returns:
        A list of mutated child strings.
    """
    _, score = parent_entry
    num_children = min(int(score * 100) + 1, MAX_CHILDREN_PER_PARENT)

    children = []
    for _ in range(num_children):
        partner_idx = random.randint(0, NUM_SELECTED)
        partner = population_scores[partner_idx][0]

        child1, child2 = crossover(parent_entry[0], partner)
        children.append(mutate(child1, genes))
        children.append(mutate(child2, genes))

    return children


def generate_random_individual(length: int, genes: list[str]) -> str:
    """Generate a random string of the given length from the gene pool.

    Args:
        length: The desired string length.
        genes: The list of valid characters.

    Returns:
        A randomly generated string.
    """
    return "".join(random.choice(genes) for _ in range(length))


def validate_inputs(target: str, genes: list[str]) -> None:
    """Validate that the configuration and inputs are consistent.

    Args:
        target: The target string.
        genes: The list of valid characters.

    Raises:
        ValueError: If population size is too small or target contains
                    characters not present in the gene pool.
    """
    if POPULATION_SIZE < NUM_SELECTED:
        raise ValueError(
            f"POPULATION_SIZE ({POPULATION_SIZE}) must be greater than "
            f"NUM_SELECTED ({NUM_SELECTED})."
        )

    missing = sorted(set(c for c in target if c not in genes))
    if missing:
        raise ValueError(
            f"Characters {missing} are not in the gene pool; "
            f"evolution cannot converge."
        )


def run_genetic_algorithm(
    target: str, genes: list[str], debug: bool = True
) -> tuple[int, int, str]:
    """Run a genetic algorithm to evolve a population toward the target string.

    Args:
        target: The target string to evolve toward.
        genes: The list of valid characters for generating individuals.
        debug: If True, print progress every DEBUG_INTERVAL generations.

    Returns:
        A tuple of (generation_count, total_population_seen, best_match).
    """
    validate_inputs(target, genes)

    target_length = len(target)

    # Initialize population with random individuals
    population = [
        generate_random_individual(target_length, genes)
        for _ in range(POPULATION_SIZE)
    ]

    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate fitness for each individual
        scored_population = [evaluate(individual, target) for individual in population]
        scored_population.sort(key=lambda entry: entry[1], reverse=True)

        best_individual, best_score = scored_population[0]

        # Check for convergence
        if best_individual == target:
            return (generation, total_population, best_individual)

        # Periodic debug output
        if debug and generation % DEBUG_INTERVAL == 0:
            print(f"\nGeneration: {generation}")
            print(f"Total Population: {total_population}")
            print(f"Best score: {best_score}")
            print(f"Best string: {best_individual}")

        # Carry over a portion of the current population (elitism)
        num_kept = int(POPULATION_SIZE * KEEP_RATIO)
        population = list(population[:num_kept])

        # Normalize scores by target length for selection
        normalized_scores = [
            (individual, score / target_length)
            for individual, score in scored_population
        ]

        # Breed selected parents to fill the population
        for i in range(NUM_SELECTED):
            children = select_and_breed(normalized_scores[i], normalized_scores, genes)
            population.extend(children)

            if len(population) >= POPULATION_SIZE:
                break


if __name__ == "__main__":
    target_str = (
        "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"
    )

    gene_pool = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    generation, total_pop, result = run_genetic_algorithm(target_str, gene_pool)

    print(f"\nGeneration: {generation}")
    print(f"Total Population: {total_pop}")
    print(f"Target: {result}")
```
