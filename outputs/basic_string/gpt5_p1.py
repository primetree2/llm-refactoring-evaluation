"""Simple genetic algorithm for evolving a string toward a target string."""

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
    """Evaluate how many characters in the candidate match the target.

    Args:
        candidate: The string being evaluated.
        target: The target string.

    Returns:
        A tuple containing the candidate string and its fitness score.
    """
    score = sum(
        1 for index, character in enumerate(candidate) if character == target[index]
    )
    return candidate, float(score)


def crossover(parent_a: str, parent_b: str) -> Tuple[str, str]:
    """Combine two parent strings using a single crossover point.

    Args:
        parent_a: The first parent string.
        parent_b: The second parent string.

    Returns:
        Two child strings created from the parents.
    """
    crossover_point = random.randint(0, len(parent_a) - 1)
    child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
    child_b = parent_b[:crossover_point] + parent_a[crossover_point:]
    return child_a, child_b


def mutate(individual: str, genes: Sequence[str]) -> str:
    """Mutate a string by replacing one character with a random gene.

    Note:
        The index selection intentionally preserves the original behavior,
        including the possibility of selecting `-1`, which targets the last
        character in the string.

    Args:
        individual: The string to mutate.
        genes: Available characters that may be inserted during mutation.

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
    population_scores: Sequence[PopulationScore],
    genes: Sequence[str],
) -> List[str]:
    """Create children from one parent and randomly chosen partners.

    The number of children created depends on the normalized fitness score
    of the parent and is capped at `MAX_CHILDREN_PER_PARENT`.

    Args:
        parent: A tuple containing the parent string and its normalized score.
        population_scores: The scored population used to select partners.
        genes: Available characters used during mutation.

    Returns:
        A list of newly created child strings.
    """
    new_population: List[str] = []
    score = parent[1]
    child_count = min(int(score * 100) + 1, MAX_CHILDREN_PER_PARENT)

    for _ in range(child_count):
        # Preserve the original inclusive selection range.
        partner_index = random.randint(0, N_SELECTED)
        partner = population_scores[partner_index][0]

        child_a, child_b = crossover(parent[0], partner)
        new_population.append(mutate(child_a, genes))
        new_population.append(mutate(child_b, genes))

    return new_population


def _validate_inputs(target: str, genes: Sequence[str]) -> None:
    """Validate algorithm configuration and gene availability.

    Args:
        target: The target string to evolve toward.
        genes: Available characters used to generate candidates.

    Raises:
        ValueError: If the population settings are invalid or if the target
            contains characters that are not present in the gene pool.
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


def _generate_initial_population(
    target_length: int, genes: Sequence[str]
) -> List[str]:
    """Generate the initial random population.

    Args:
        target_length: Length of each generated candidate string.
        genes: Available characters used to build the population.

    Returns:
        A list of randomly generated candidate strings.
    """
    return [
        "".join(random.choice(genes) for _ in range(target_length))
        for _ in range(N_POPULATION)
    ]


def basic(
    target: str, genes: Sequence[str], debug: bool = True
) -> Tuple[int, int, str]:
    """Run the genetic algorithm until the target string is produced.

    Args:
        target: The desired final string.
        genes: Available characters used for generation and mutation.
        debug: Whether to print progress every `DEBUG_INTERVAL` generations.

    Returns:
        A tuple containing:
            - generation count
            - total population processed
            - evolved target string
    """
    _validate_inputs(target, genes)

    population = _generate_initial_population(len(target), genes)
    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate and rank the current population.
        population_scores = [evaluate(individual, target) for individual in population]
        population_scores.sort(key=lambda item: item[1], reverse=True)

        best = population_scores[0]

        if best[0] == target:
            return generation, total_population, best[0]

        if debug and generation % DEBUG_INTERVAL == 0:
            print("\nGeneration:", generation)
            print("Total Population:", total_population)
            print("Best score:", best[1])
            print("Best string:", best[0])

        # Preserve the original behavior by keeping the first third of the
        # current population in its existing order.
        keep_count = N_POPULATION // 3
        population = population[:keep_count]

        # Normalize scores before selection.
        population_scores = [
            (candidate, score / len(target))
            for candidate, score in population_scores
        ]

        # Generate children from the selected portion of the population.
        for index in range(N_SELECTED):
            children = select(population_scores[index], population_scores, genes)
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

    generation, total_population, target = basic(target_str, genes_list)

    print("\nGeneration:", generation)
    print("Total Population:", total_population)
    print("Target:", target)
