"""
Genetic algorithm implementation to evolve a string towards a target.
"""

import random
from typing import List, Sequence, Tuple

# Global configuration constants (preserved from original naming where possible)
N_POPULATION = 200
N_SELECTED = 50
MUTATION_PROBABILITY = 0.4
MAX_CHILDREN_PER_PARENT = 10
DEBUG_INTERVAL = 10

# Type alias for readability
PopulationScore = Tuple[str, float]


def evaluate(a: str, b: str) -> PopulationScore:
    """
    Evaluate similarity between two strings by counting matching characters.

    Args:
        a: Candidate string.
        b: Target string.

    Returns:
        Tuple containing the candidate and its fitness score.
    """
    s = sum(1 for i in range(len(a)) if a[i] == b[i])
    return (a, float(s))


def crossover(x: str, y: str) -> Tuple[str, str]:
    """
    Perform single-point crossover between two parent strings.

    Args:
        x: First parent string.
        y: Second parent string.

    Returns:
        Two child strings resulting from the crossover.
    """
    p = random.randint(0, len(x) - 1)
    c1 = x[:p] + y[p:]
    c2 = y[:p] + x[p:]
    return (c1, c2)


def mutate(z: str, genes: Sequence[str]) -> str:
    """
    Mutate a string at a random position with a given probability.

    Note: Original index calculation (randint(0, len) - 1) is preserved,
    which can result in position -1 (last character).

    Args:
        z: The individual string to mutate.
        genes: Available characters for mutation.

    Returns:
        The (potentially mutated) string.
    """
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        l = list(z)
        pos = random.randint(0, len(z)) - 1
        g = random.choice(genes)
        l[pos] = g
        return "".join(l)
    return z


def select(p1: PopulationScore, population_score: List[PopulationScore], genes: Sequence[str]) -> List[str]:
    """
    Generate children from a parent by breeding with randomly selected partners.

    Args:
        p1: Primary parent entry (string, score).
        population_score: List of scored individuals for partner selection.
        genes: Available characters.

    Returns:
        List of mutated child strings.
    """
    newpop: List[str] = []
    score = p1[1]
    child_n = min(int(score * 100) + 1, MAX_CHILDREN_PER_PARENT)

    for _ in range(child_n):
        idx = random.randint(0, N_SELECTED)
        parent2 = population_score[idx][0]

        kids = crossover(p1[0], parent2)
        c1, c2 = kids

        m1 = mutate(c1, genes)
        m2 = mutate(c2, genes)

        newpop.append(m1)
        newpop.append(m2)

    return newpop


def basic(target: str, genes: Sequence[str], debug: bool = True) -> Tuple[int, int, str]:
    """
    Run the genetic algorithm until the target string is evolved.

    Args:
        target: The target string to match.
        genes: List of allowed characters.
        debug: Whether to print debug information periodically.

    Returns:
        Tuple of (generations, total population evaluated, best match).
    """
    if N_POPULATION < N_SELECTED:
        raise ValueError(str(N_POPULATION) + " must be bigger than " + str(N_SELECTED))

    # Check for missing genes
    missing = []
    for c in target:
        if c not in genes:
            if c not in missing:
                missing.append(c)

    if len(missing) > 0:
        raise ValueError(str(missing) + " is not in genes list, evolution cannot converge")

    # Generate initial random population
    population: List[str] = []
    for _ in range(N_POPULATION):
        s = "".join(random.choice(genes) for _ in range(len(target)))
        population.append(s)

    generation = 0
    total_population = 0

    while True:
        generation += 1
        total_population += len(population)

        # Evaluate fitness of entire population
        population_score: List[PopulationScore] = [evaluate(item, target) for item in population]
        population_score.sort(key=lambda x: x[1], reverse=True)

        best = population_score[0]

        if best[0] == target:
            return (generation, total_population, best[0])

        if debug and generation % DEBUG_INTERVAL == 0:
            print("\nGeneration:", generation)
            print("Total Population:", total_population)
            print("Best score:", best[1])
            print("Best string:", best[0])

        # Keep top 1/3 of current population (elitism)
        keep = N_POPULATION // 3
        population = population[:keep]

        # Normalize scores for selection
        population_score = [(it[0], it[1] / len(target)) for it in population_score]

        # Breed new individuals from selected parents
        for i in range(N_SELECTED):
            children = select(population_score[i], population_score, genes)
            population.extend(children)

            if len(population) > N_POPULATION:
                break


if __name__ == "__main__":
    # Set random seed as in original (pointless but preserved for identical behavior)
    r = random.randint(0, 1000)
    random.seed(r)

    target_str = "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"

    genes_list = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    result = basic(target_str, genes_list)

    generation = result[0]
    population = result[1]
    target = result[2]

    print("\nGeneration:", generation)
    print("Total Population:", population)
    print("Target:", target)
