# population = list[chromosome]
# chromosome = list[gene] = list of mitiq executables
# gene = parameterized error mitigator
# assume gene is executable, takes only circuit as parameter


def ideal(circuit):
    return execute(circuit)

def noisy(circuit, noise):
    return execute(circuit, noise)

def mitigated(chromosome, circuit, noise):
    result = None
    for executable in chromosome:
        mitigate(executable, circuit, noise) # TODO: chain the results
    return result
    

def evaluate_fitness(chromosome, circuit):
    """
    Evaluates the mitigation performance of 'chromosome' on 'circuit'
    """
    fitness = 0
    ideal_measurement = ideal(circuit)
    noisy_measurement = noisy(circuit)
    mitigated_measurement = mitigated(chromosome, circuit)
    # TODO: fitness = relative gain in mitigation
    # higher fitness = the difference between noisy and ideal > mitigated and ideal
    # ideal noise is as far away as possible
    fitness = 1 - diff(mitigated_measurement - ideal_measurement) / diff(noisy_measurement - ideal_measurement)
    return fitness


def mutate(chromosome, p=0.5):
    if random.uniform(0,1) < p: # some p
        return new(chromosome) # or change its parameter
    return chromosome


def grow_shrink(population, p=0.5):
    return population
    # TODO: implement
    if random.uniform(0,1) < p:
        pass # pop random individual
    if random.uniform(0,1) < p:
        pass # add random individual
    return population


def crossover(population, p=0.5):
    return population
    # TODO: implement
    # for each pair: maybe crossover
    for i in range(len(population) // 2):
        if random.uniform(0,1) < p: # TODO: select by random number
            pass
            # partition 2 chromosomes


def genetic_algorithm(circuit, generations=5, population_size=10):
    population = initialize_population(population_size)
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(chromosome, circuit) for chromosome in population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        print(avg_fitness) # TODO: nice plot

        # Selection
        # Simple selection strategy: sort by fitness and select top half
        population = [chromosome for _, chromosome in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        parents = population[:len(population)//2]

        population = grow_shrink(population)
        population = [mutate(chromosome) for chromosome in population]
        population = crossover(population)

        print(f"Generation {generation}, Best Score: {max(fitness_scores)}")

    return population
    # Return the best individual and its score
    # best_index = np.argmax(fitness_scores)
    # return population[best_index], fitness_scores[best_index]

circuit = None # TODO: some benchmarking circuit
print(genetic_algorithm(circuit))
