"""
A genetic algorithm to evolve a population of random strings toward a target string.
"""

import random
from typing import List, Sequence, Tuple

# Configuration Constants
POPULATION_SIZE = 200
NUM_SELECTED = 50
MUTATION_PROBABILITY = 0.4
MAX_CHILDREN_PER_PARENT = 10
LOGGING_INTERVAL = 10

# Preserve the original module-level seeding behavior
random.seed(random.randint(0, 1000))

# Type aliases for clarity
PopulationScore = Tuple[str, float]


def evaluate_fitness(candidate: str, target: str) -> PopulationScore:
    """
    Evaluate the similarity between a candidate string and the target.

    Args:
        candidate: The string to evaluate.
        target: The goal string.

    Returns:
        A tuple of (candidate_string, matching_character_count).
    """
    score = sum(1 for c, t in zip(candidate, target) if c == t)
    return candidate, float(score)


def crossover(parent1: str, parent2: str) -> Tuple[str, str]:
    """
    Perform single-point crossover between two parent strings.

    Args:
        parent1: The first parent string.
        parent2: The second parent string.

    Returns:
        A tuple containing two generated child strings.
    """
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual: str, genes: Sequence[str]) -> str:
    """
    Mutate a random position in the individual with a given probability.

    Note:
        Intentionally preserves the original index logic which allows 
        the position to be -1 (targeting the last character).

    Args:
        individual: The string to potentially mutate.
        genes: The list of valid characters for mutation.

    Returns:
        The mutated string, or the original if no mutation occurred.
    """
    if random.random() < MUTATION_PROBABILITY:
        chars = list(individual)
        position = random.randint(0, len(individual)) - 1
        chars[position] = random.choice(genes)
        return "".join(chars)
    return individual


def select_and_breed(
    parent_entry: PopulationScore,
    population_scores: List[PopulationScore],
    genes: Sequence[str],
) -> List[str]:
    """
    Generate children by pairing a parent with random partners from the population.

    The number of children is proportional to the parent's normalized fitness.

    Args:
        parent_entry: A tuple of (parent_string, normalized_fitness).
        population_scores: The ranked population used to select partners.
        genes: The pool of characters available for mutation.

    Returns:
        A list of newly generated, potentially mutated child strings.
    """
    parent_string, normalized_score = parent_entry
    num_children = min(int(normalized_score * 100) + 1, MAX_CHILDREN_PER_PARENT)

    children = []
    for _ in range(num_children):
        # The inclusive upper bound matches the original random.randint logic
        partner_idx = random.randint(0, NUM_SELECTED)
        partner_string = population_scores[partner_idx][0]

        child1, child2 = crossover(parent_string, partner_string)

        children.append(mutate(child1, genes))
        children.append(mutate(child2, genes))

    return children


def run_evolution(
    target: str, genes: Sequence[str], debug: bool = True
) -> Tuple[int, int, str]:
    """
    Execute the genetic algorithm to evolve strings until the target is met.

    Args:
        target: The desired string to evolve toward.
        genes: The list of valid characters to construct strings from.
        debug: If True, prints progress details every LOGGING_INTERVAL generations.

    Returns:
        A tuple containing (total_generations, total_population_evaluated, best_string).
        
    Raises:
        ValueError: If population settings are invalid or the target contains
            characters missing from the gene pool.
    """
    if POPULATION_SIZE < NUM_SELECTED:
        raise ValueError(f"{POPULATION_SIZE} must be bigger than {NUM_SELECTED}")

    # Validate that all required characters are present in the gene pool
    missing_genes = []
    for char in target:
        if char not in genes and char not in missing_genes:
            missing_genes.append(char)

    if missing_genes:
        raise ValueError(
            f"{missing_genes} is not in genes list, evolution cannot converge"
        )

    target_length = len(target)

    # Generate initial random population
    population = [
        "".join(random.choice(genes) for _ in range(target_length))
        for _ in range(POPULATION_SIZE)
    ]

    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate and rank the population
        population_scores = [evaluate_fitness(ind, target) for ind in population]
        population_scores.sort(key=lambda item: item[1], reverse=True)

        best_string, best_score = population_scores[0]

        # Check for convergence
        if best_string == target:
            return generation, total_population, best_string

        if debug and generation % LOGGING_INTERVAL == 0:
            print(f"\nGeneration: {generation}")
            print(f"Total Population: {total_population}")
            print(f"Best score: {best_score}")
            print(f"Best string: {best_string}")

        # Carry over a portion of the original population (Elitism).
        # Note: Preserving the original behavior where the unsorted 'population' list is sliced.
        keep_count = POPULATION_SIZE // 3
        population = population[:keep_count]

        # Normalize the fitness scores by the target length
        normalized_scores = [
            (ind, score / target_length) for ind, score in population_scores
        ]

        # Generate children from the selected pool to replenish the population
        for i in range(NUM_SELECTED):
            children = select_and_breed(normalized_scores[i], normalized_scores, genes)
            population.extend(children)

            if len(population) > POPULATION_SIZE:
                break


if __name__ == "__main__":
    target_str = (
        "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"
    )

    genes_list = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    final_generation, total_evaluated, evolved_string = run_evolution(
        target_str, genes_list
    )

    print(f"\nGeneration: {final_generation}")
    print(f"Total Population: {total_evaluated}")
    print(f"Target: {evolved_string}")
