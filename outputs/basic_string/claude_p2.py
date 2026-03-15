"""
A simple genetic algorithm that evolves a population of random strings
toward a target string through evaluation, crossover, mutation, and selection.
"""

import random
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
POPULATION_SIZE = 200
NUM_SELECTED = 50
MUTATION_PROBABILITY = 0.4
MAX_CHILDREN_PER_PARENT = 10
ELITISM_FRACTION = 1 / 3
LOG_INTERVAL = 10

# Seed the RNG (preserves original seeding behaviour)
random.seed(random.randint(0, 1000))

# Type aliases for clarity
ScoredIndividual = Tuple[str, float]


def evaluate_fitness(candidate: str, target: str) -> ScoredIndividual:
    """Return *candidate* paired with the number of positions matching *target*.

    Args:
        candidate: The string to score.
        target: The reference / goal string.

    Returns:
        A tuple of (candidate, matching_character_count).
    """
    score = sum(c == t for c, t in zip(candidate, target))
    return (candidate, float(score))


def single_point_crossover(parent_a: str, parent_b: str) -> Tuple[str, str]:
    """Perform single-point crossover on two equal-length parent strings.

    A random crossover point is chosen; characters before that point come
    from one parent, and characters from that point onward come from the
    other.

    Returns:
        Two child strings.
    """
    point = random.randint(0, len(parent_a) - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def mutate(individual: str, genes: Sequence[str]) -> str:
    """With probability ``MUTATION_PROBABILITY``, replace one random character.

    Args:
        individual: The string that may be mutated.
        genes: Pool of characters eligible for substitution.

    Returns:
        The (possibly mutated) string.
    """
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        chars = list(individual)
        # Original logic: randint(0, len) - 1  →  index in [-1, len-1]
        position = random.randint(0, len(individual)) - 1
        chars[position] = random.choice(genes)
        return "".join(chars)
    return individual


def breed_children(
    parent_entry: ScoredIndividual,
    population_scores: List[ScoredIndividual],
    genes: Sequence[str],
) -> List[str]:
    """Produce children by crossing *parent_entry* with random partners.

    The number of breeding rounds is proportional to the parent's
    normalised fitness score, capped at ``MAX_CHILDREN_PER_PARENT``.

    Args:
        parent_entry: (string, normalised_score) for the primary parent.
        population_scores: Scored population used to pick random partners.
        genes: Character pool passed through to :func:`mutate`.

    Returns:
        A list of mutated child strings.
    """
    normalised_score = parent_entry[1]
    num_rounds = min(int(normalised_score * 100) + 1, MAX_CHILDREN_PER_PARENT)

    children: List[str] = []
    for _ in range(num_rounds):
        # Pick a random partner from the top selected individuals
        partner_idx = random.randint(0, NUM_SELECTED)
        partner = population_scores[partner_idx][0]

        child_a, child_b = single_point_crossover(parent_entry[0], partner)
        children.append(mutate(child_a, genes))
        children.append(mutate(child_b, genes))

    return children


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_config() -> None:
    """Raise if the population size is smaller than the selection count."""
    if POPULATION_SIZE < NUM_SELECTED:
        raise ValueError(
            f"{POPULATION_SIZE} must be bigger than {NUM_SELECTED}"
        )


def _validate_genes(target: str, genes: Sequence[str]) -> None:
    """Raise if *target* contains characters absent from *genes*."""
    missing = list(dict.fromkeys(c for c in target if c not in genes))
    if missing:
        raise ValueError(
            f"{missing} is not in genes list, evolution cannot converge"
        )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def run_evolution(
    target: str,
    genes: Sequence[str],
    debug: bool = True,
) -> Tuple[int, int, str]:
    """Evolve a random population of strings until *target* is matched.

    Args:
        target: The goal string the algorithm tries to reproduce.
        genes: The set of characters available for building strings.
        debug: If ``True``, print progress every ``LOG_INTERVAL`` generations.

    Returns:
        A tuple of (generation_number, total_individuals_evaluated, best_string).
    """
    _validate_config()
    _validate_genes(target, genes)

    target_len = len(target)

    # Generate the initial random population
    population: List[str] = [
        "".join(random.choice(genes) for _ in range(target_len))
        for _ in range(POPULATION_SIZE)
    ]

    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate and rank every individual
        population_scores: List[ScoredIndividual] = [
            evaluate_fitness(individual, target) for individual in population
        ]
        population_scores.sort(key=lambda entry: entry[1], reverse=True)

        best_individual, best_score = population_scores[0]

        # Termination check
        if best_individual == target:
            return (generation, total_population, best_individual)

        # Periodic debug logging
        if debug and generation % LOG_INTERVAL == 0:
            print(f"\nGeneration: {generation}")
            print(f"Total Population: {total_population}")
            print(f"Best score: {best_score}")
            print(f"Best string: {best_individual}")

        # Elitism: carry forward a fraction of the current population
        keep_count = int(POPULATION_SIZE * ELITISM_FRACTION)
        population = population[:keep_count]

        # Normalise scores by target length for selection weighting
        normalised_scores: List[ScoredIndividual] = [
            (ind, score / target_len) for ind, score in population_scores
        ]

        # Selection and breeding loop
        for i in range(NUM_SELECTED):
            children = breed_children(normalised_scores[i], normalised_scores, genes)
            population.extend(children)

            if len(population) > POPULATION_SIZE:
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TARGET = (
        "This is a genetic algorithm to evaluate, combine, "
        "evolve, and mutate a string!"
    )

    GENES = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    gen, total_pop, matched = run_evolution(TARGET, GENES)

    print(f"\nGeneration: {gen}")
    print(f"Total Population: {total_pop}")
    print(f"Target: {matched}")
