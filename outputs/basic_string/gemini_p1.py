"""
Genetic algorithm implementation to evolve a string towards a target.
"""

import random
from typing import List, Sequence, Tuple

# Global configuration constants
POPULATION_SIZE = 200
NUM_SELECTED = 50
MUTATION_PROBABILITY = 0.4
MAX_CHILDREN_PER_ITERATION = 10
DEBUG_INTERVAL = 10

# Type aliases for readability
Individual = str
FitnessScore = float
PopulationScore = Tuple[Individual, FitnessScore]


def evaluate(candidate: Individual, target: str) -> PopulationScore:
    """
    Evaluate the similarity between a candidate string and the target.

    Args:
        candidate: The string to be evaluated.
        target: The target string.

    Returns:
        A tuple of the candidate string and its similarity score 
        (number of matching characters).
    """
    score = sum(1 for c, t in zip(candidate, target) if c == t)
    return candidate, float(score)


def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Perform single-point crossover between two parent strings.

    Args:
        parent1: The first parent string.
        parent2: The second parent string.

    Returns:
        A tuple containing two child strings.
    """
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual: Individual, genes: Sequence[str]) -> Individual:
    """
    Mutate a random character in the string with a given probability.

    Args:
        individual: The string to potentially mutate.
        genes: The list of valid characters for mutation.

    Returns:
        The mutated string or the original string if no mutation occurred.
    """
    if random.random() < MUTATION_PROBABILITY:
        chars = list(individual)
        # Note: The original logic allows -1 which correctly modifies the last element
        pos = random.randint(0, len(individual)) - 1
        chars[pos] = random.choice(genes)
        return "".join(chars)
    return individual


def select_and_breed(
    parent_entry: PopulationScore,
    population_scores: List[PopulationScore],
    genes: Sequence[str],
) -> List[Individual]:
    """
    Select partners for a parent and generate mutated offspring.

    Args:
        parent_entry: A tuple of the parent string and its normalized fitness score.
        population_scores: The list of candidate parents and their scores.
        genes: The list of valid characters for mutation.

    Returns:
        A list of child strings generated from the parent and random partners.
    """
    parent_string, normalized_score = parent_entry
    num_iterations = min(int(normalized_score * 100) + 1, MAX_CHILDREN_PER_ITERATION)

    children = []
    for _ in range(num_iterations):
        # Includes upper bound to mimic original logic
        partner_idx = random.randint(0, NUM_SELECTED)
        partner_string = population_scores[partner_idx][0]

        child1, child2 = crossover(parent_string, partner_string)
        
        children.append(mutate(child1, genes))
        children.append(mutate(child2, genes))

    return children


def run_genetic_algorithm(
    target: str, genes: Sequence[str], debug: bool = True
) -> Tuple[int, int, Individual]:
    """
    Execute the genetic algorithm to evolve a population towards the target string.

    Args:
        target: The target string to evolve.
        genes: The available characters to use for strings.
        debug: If True, prints progress periodically.

    Returns:
        A tuple containing the number of generations, the total population generated, 
        and the final matched string.
    """
    if POPULATION_SIZE < NUM_SELECTED:
        raise ValueError(f"{POPULATION_SIZE} must be bigger than {NUM_SELECTED}")

    missing_genes = []
    for char in target:
        if char not in genes and char not in missing_genes:
            missing_genes.append(char)

    if missing_genes:
        raise ValueError(f"{missing_genes} is not in genes list, evolution cannot converge")

    target_length = len(target)
    
    # Initialize the first generation
    population: List[Individual] = [
        "".join(random.choice(genes) for _ in range(target_length))
        for _ in range(POPULATION_SIZE)
    ]

    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate and sort the population by fitness
        population_scores = [evaluate(individual, target) for individual in population]
        population_scores.sort(key=lambda x: x[1], reverse=True)

        best_individual, best_score = population_scores[0]

        if best_individual == target:
            return generation, total_population, best_individual

        if debug and generation % DEBUG_INTERVAL == 0:
            print(f"\nGeneration: {generation}")
            print(f"Total Population: {total_population}")
            print(f"Best score: {best_score}")
            print(f"Best string: {best_individual}")

        # Keep a fraction of the current population (Elitism preservation)
        keep_count = POPULATION_SIZE // 3
        population = population[:keep_count]

        # Normalize fitness scores based on target length
        normalized_scores = [
            (ind, score / target_length) for ind, score in population_scores
        ]

        # Breed new children to fill the population
        for i in range(NUM_SELECTED):
            children = select_and_breed(normalized_scores[i], normalized_scores, genes)
            population.extend(children)

            if len(population) > POPULATION_SIZE:
                break


if __name__ == "__main__":
    target_string = "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"

    genes_pool = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    final_generation, total_pop, result_target = run_genetic_algorithm(target_string, genes_pool)

    print(f"\nGeneration: {final_generation}")
    print(f"Total Population: {total_pop}")
    print(f"Target: {result_target}")
