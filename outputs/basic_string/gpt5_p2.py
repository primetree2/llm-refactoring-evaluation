"""Genetic algorithm for evolving a string toward a target value."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

# Genetic algorithm configuration.
N_POPULATION = 200
N_SELECTED = 50
MUTATION_PROBABILITY = 0.4
MAX_CHILDREN_PER_PARENT = 10
DEBUG_INTERVAL = 10

# Preserve the original module-level seeding behavior.
random.seed(random.randint(0, 1000))

PopulationScore = Tuple[str, float]


def evaluate(candidate: str, target: str) -> PopulationScore:
    """Evaluate similarity between a candidate string and the target.

    The score is the number of positions with matching characters.

    Args:
        candidate: String being evaluated.
        target: Target string.

    Returns:
        A tuple containing the candidate string and its fitness score.
    """
    score = sum(
        1 for index, character in enumerate(candidate) if character == target[index]
    )
    return candidate, float(score)


def crossover(parent_a: str, parent_b: str) -> Tuple[str, str]:
    """Combine two strings using single-point crossover.

    Args:
        parent_a: First parent string.
        parent_b: Second parent string.

    Returns:
        Two children generated from the crossover point.
    """
    crossover_point = random.randint(0, len(parent_a) - 1)
    child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
    child_b = parent_b[:crossover_point] + parent_a[crossover_point:]
    return child_a, child_b


def mutate(individual: str, genes: Sequence[str]) -> str:
    """Mutate one random character in a string with a fixed probability.

    Note:
        The index calculation intentionally preserves the original behavior:
        ``random.randint(0, len(individual)) - 1`` may produce ``-1``,
        which targets the last character.

    Args:
        individual: String to mutate.
        genes: Available characters for mutation.

    Returns:
        The mutated string, or the original string if no mutation occurs.
    """
    characters = list(individual)

    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        position = random.randint(0, len(individual)) - 1
        characters[position] = random.choice(genes)

    return "".join(characters)


def select(
    parent: PopulationScore,
    population_score: Sequence[PopulationScore],
    genes: Sequence[str],
) -> List[str]:
    """Generate children for a parent by pairing it with random partners.

    The number of child-generation rounds depends on the normalized score
    and is capped at ``MAX_CHILDREN_PER_PARENT``.

    Args:
        parent: A tuple of (individual, normalized_score).
        population_score: Scored population used for partner selection.
        genes: Available characters for mutation.

    Returns:
        A list of newly generated child strings.
    """
    new_population: List[str] = []
    score = parent[1]
    child_count = min(int(score * 100) + 1, MAX_CHILDREN_PER_PARENT)

    for _ in range(child_count):
        # Preserve the original inclusive upper bound.
        partner_index = random.randint(0, N_SELECTED)
        partner = population_score[partner_index][0]

        child_a, child_b = crossover(parent[0], partner)
        new_population.append(mutate(child_a, genes))
        new_population.append(mutate(child_b, genes))

    return new_population


def _validate_inputs(target: str, genes: Sequence[str]) -> None:
    """Validate algorithm configuration and input data.

    Args:
        target: Desired target string.
        genes: Allowed characters.

    Raises:
        ValueError: If configuration is invalid or target contains
            characters not present in the gene pool.
    """
    if N_POPULATION < N_SELECTED:
        raise ValueError(f"{N_POPULATION} must be bigger than {N_SELECTED}")

    missing: List[str] = []
    for character in target:
        if character not in genes and character not in missing:
            missing.append(character)

    if missing:
        raise ValueError(
            f"{missing} is not in genes list, evolution cannot converge"
        )


def _generate_initial_population(target_length: int, genes: Sequence[str]) -> List[str]:
    """Create the initial random population.

    Args:
        target_length: Length of each generated individual.
        genes: Allowed characters.

    Returns:
        A list of random strings.
    """
    return [
        "".join(random.choice(genes) for _ in range(target_length))
        for _ in range(N_POPULATION)
    ]


def basic(target: str, genes: Sequence[str], debug: bool = True) -> Tuple[int, int, str]:
    """Run the genetic algorithm until the target string is found.

    Args:
        target: Target string to evolve toward.
        genes: Allowed characters used for generation and mutation.
        debug: Whether to print progress information every few generations.

    Returns:
        A tuple containing:
            - generation count
            - total population processed over time
            - the evolved target string
    """
    _validate_inputs(target, genes)

    population = _generate_initial_population(len(target), genes)
    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate and rank the current population.
        population_score = [evaluate(item, target) for item in population]
        population_score.sort(key=lambda item: item[1], reverse=True)

        best = population_score[0]

        if best[0] == target:
            return generation, total_population, best[0]

        if debug and generation % DEBUG_INTERVAL == 0:
            print("\nGeneration:", generation)
            print("Total Population:", total_population)
            print("Best score:", best[1])
            print("Best string:", best[0])

        # Preserve the original behavior: keep the first third of the
        # existing population in its current order.
        keep_count = N_POPULATION // 3
        population = population[:keep_count]

        # Normalize scores before selection.
        population_score = [
            (individual, score / len(target))
            for individual, score in population_score
        ]

        # Generate children from the selected portion of the population.
        for index in range(N_SELECTED):
            children = select(population_score[index], population_score, genes)
            population.extend(children)

            if len(population) > N_POPULATION:
                break


if __name__ == "__main__":
    target_str = (
        "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"
    )

    genes_list = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    generation, population, target = basic(target_str, genes_list)

    print("\nGeneration:", generation)
    print("Total Population:", population)
    print("Target:", target)
