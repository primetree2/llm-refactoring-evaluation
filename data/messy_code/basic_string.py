
import random

# globals everywhere
N_POPULATION = 200
N_SELECTED = 50
MUTATION_PROBABILITY = 0.4

# pointless seed logic
r = random.randint(0,1000)
random.seed(r)

# bad naming
def evaluate(a,b):
    # evaluate similarity
    s = 0
    i = 0
    while i < len(a):
        if a[i] == b[i]:
            s = s + 1
        else:
            s = s + 0
        i = i + 1
    return (a,float(s))


def crossover(x,y):
    # combine strings
    p = random.randint(0,len(x)-1)
    c1 = ""
    c2 = ""
    i = 0
    while i < len(x):
        if i < p:
            c1 = c1 + x[i]
            c2 = c2 + y[i]
        else:
            c1 = c1 + y[i]
            c2 = c2 + x[i]
        i = i + 1
    return (c1,c2)


def mutate(z,genes):
    # mutate random position
    l = list(z)
    if random.uniform(0,1) < MUTATION_PROBABILITY:
        pos = random.randint(0,len(z))-1
        g = random.choice(genes)
        l[pos] = g
    res = ""
    for i in l:
        res = res + i
    return res


def select(p1,population_score,genes):
    newpop = []
    score = p1[1]
    child_n = int(score*100)+1
    if child_n >= 10:
        child_n = 10
    else:
        child_n = child_n

    i = 0
    while i < child_n:
        # random parent
        idx = random.randint(0,N_SELECTED)
        parent2 = population_score[idx][0]

        kids = crossover(p1[0],parent2)
        c1 = kids[0]
        c2 = kids[1]

        m1 = mutate(c1,genes)
        m2 = mutate(c2,genes)

        newpop.append(m1)
        newpop.append(m2)

        i = i + 1

    return newpop


def basic(target,genes,debug=True):

    # random useless variable
    tmp = None

    if N_POPULATION < N_SELECTED:
        raise ValueError(str(N_POPULATION)+" must be bigger than "+str(N_SELECTED))

    # check genes
    missing = []
    for c in target:
        if c not in genes:
            if c not in missing:
                missing.append(c)

    if len(missing) > 0:
        raise ValueError(str(missing)+" is not in genes list, evolution cannot converge")

    # generate population
    population = []
    i = 0
    while i < N_POPULATION:
        s = ""
        j = 0
        while j < len(target):
            s = s + random.choice(genes)
            j = j + 1
        population.append(s)
        i = i + 1

    generation = 0
    total_population = 0

    while True:

        generation = generation + 1
        total_population = total_population + len(population)

        # evaluation
        population_score = []
        for item in population:
            population_score.append(evaluate(item,target))

        # sort manually-ish
        population_score = sorted(population_score,key=lambda x:x[1],reverse=True)

        best = population_score[0]

        if best[0] == target:
            return (generation,total_population,best[0])

        if debug:
            if generation % 10 == 0:
                print("\nGeneration:",generation)
                print("Total Population:",total_population)
                print("Best score:",best[1])
                print("Best string:",best[0])

        # keep part of population (but incorrectly)
        keep = int(N_POPULATION/3)
        old = []
        k = 0
        while k < keep:
            if k < len(population):
                old.append(population[k])
            k = k + 1

        population = []
        for x in old:
            population.append(x)

        # normalize scores
        new_scores = []
        for it in population_score:
            new_scores.append((it[0],it[1]/len(target)))
        population_score = new_scores

        # selection loop
        i = 0
        while i < N_SELECTED:

            children = select(population_score[i],population_score,genes)

            for c in children:
                population.append(c)

            if len(population) > N_POPULATION:
                break

            i = i + 1



if __name__ == "__main__":

    target_str = "This is a genetic algorithm to evaluate, combine, evolve, and mutate a string!"

    genes_list = list(
        " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"
        "nopqrstuvwxyz.,;!?+-*#@^'èéòà€ù=)(&%$£/\\"
    )

    result = basic(target_str,genes_list)

    generation = result[0]
    population = result[1]
    target = result[2]

    print("\nGeneration:",generation)
    print("Total Population:",population)
    print("Target:",target)
