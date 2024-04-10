# population = list[chromosome]
# chromosome = list[gene] = list of mitiq executables
# gene = parameterized error mitigator
# assume gene is executable, takes only circuit as parameter

from abc import ABC, abstractmethod
from functools import partial
from mitiq.benchmarks import generate_rb_circuits, generate_ghz_circuit, generate_w_circuit
from mitiq import rem, zne, ddd, Observable, PauliString, MeasurementResult, raw
import cirq
import numpy as np
from numpy import random as random

from tqdm import tqdm, trange

N_QUBITS = 5
OBS = Observable(PauliString("Z" * N_QUBITS)) # THIS IS IN THE WRONG PLACE
np.random.seed(42) # TODO: global random seed is not respected

# TODO: for now I'm just going to hardcode the genes
class BaseGene(ABC):
    def __str__(self):
        return f'base'

    def __repr__(self):
        return str(self)
    
    def __init__(self):
        pass

    @abstractmethod
    def executor(self, executable):
        pass
    
class REMGene(BaseGene):
    def __str__(self):
        return f'rem({self.p0:.2f}, {self.p1:.2f})'
        
    def __init__(self, p0=0.05, p1=0.05):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
    
    def executor(self, executable):
        icm = rem.generate_inverse_confusion_matrix(N_QUBITS, self.p0, self.p1)

        return rem.mitigate_executor(executable, inverse_confusion_matrix=icm)

class ZNEGene(BaseGene):
    def __str__(self):
        return f'zne({self.factory.__class__.__name__}, {self.scale_noise.__name__})'
    
    def __init__(self, factory, scale_noise, num_to_avg):
        super().__init__()
        self.factory = factory
        self.scale_noise = scale_noise
        self.num_to_avg = num_to_avg

    def executor(self, executable):
        return zne.mitigate_executor(
            executable,
            observable=OBS,
            factory=self.factory,
            scale_noise=self.scale_noise,
            num_to_average=self.num_to_avg
        )
    
class DDDGene(BaseGene):
    def __str__(self):
        return f'ddd({self.rule.__name__})'
        
    def __init__(self, rule):
        super().__init__()
        self.rule = rule

    def executor(self, executable):
        return ddd.mitigate_executor(
            executable,
            rule=self.rule,
            observable=OBS
        )

# TODO: this is hardcoded
def execute(circuit: cirq.Circuit, noise_level: float = 0.002, p0: float = 0.05) -> MeasurementResult:
    """Execute a circuit with depolarizing noise of strength ``noise_level`` and readout errors ...
    """
    measurements = circuit[-1]
    circuit =  circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))
    circuit.append(cirq.bit_flip(p0).on_each(circuit.all_qubits()))
    circuit.append(measurements)

    simulator = cirq.DensityMatrixSimulator()

    result = simulator.run(circuit, repetitions=1000)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def ideal(circuit):
    return raw.execute(circuit, partial(execute, noise_level=0, p0=0), OBS) # TODO: not use this raw thing

def noisy(circuit):
    return raw.execute(circuit, execute, OBS)

def mitigated(chromosome, circuit, execute):
    """
    runs circuit with chromosome mitigation chain
    exec: an executable that takes a circuit as input and outputs a QuantumResult
    """

    chain_executor = execute

    for gene in chromosome:
        chain_executor = gene.executor(chain_executor)

    result = chain_executor(circuit)

    return result
    

def evaluate_fitness(chromosome, circuit):
    """
    Evaluates the mitigation performance of 'chromosome' on 'circuit'
    """
    fitness = 0
    ideal_measurement = ideal(circuit)
    noisy_measurement = noisy(circuit)
    mitigated_measurement = mitigated(chromosome, circuit, execute)
    # TODO: fitness = relative gain in mitigation
    # higher fitness = the difference between noisy and ideal > mitigated and ideal
    # ideal noise is as far away as possible
    
    distance_mitigated = abs(mitigated_measurement - ideal_measurement)
    distance_noisy     = abs(noisy_measurement     - ideal_measurement)

    # fix for divide by zero in fitness :(
    fitness = 1 - distance_mitigated / (1e-8 + distance_noisy)
    
    return fitness


def mutate(chromosome, p=0.5):
    i = random.randint(len(chromosome))
    match chromosome[i]:
        case REMGene(p0=p0, p1=p1):
            match random.randint(3):
                case 0:
                    pass
                case 1:
                    chromosome[i].p0 += 0.01 * random.randn()
                case 2:
                    chromosome[i].p1 += 0.01 * random.randn()
            chromosome[i].p0 = np.clip(chromosome[i].p0, 0, 1)
            chromosome[i].p1 = np.clip(chromosome[i].p1, 0, 1)
        case ZNEGene():
            pass
        case DDDGene():
            pass
    return chromosome


def grow_shrink(population, p=0.5):
    return population
    # TODO: implement
    if random.uniform(0,1) < p:
        pass # pop random individual
    if random.uniform(0,1) < p:
        pass # add random individual
    return population


def crossover(population, times=None):
    if times is None:
        times = len(population) // 2

    # individual crossover
    def cross(x, y):
        ix = random.randint(len(x))
        iy = random.randint(len(y))
        # right now, we can only swap matching gene types
        if type(x[ix]) == type(y[iy]):
            t     = x[ix]
            x[ix] = y[iy]
            y[iy] = t
    
    # do an arbitrary number of crossover iterations
    for _ in range(times):
        ix, iy = random.randint(len(population), size=2)
        cross(population[ix], population[iy])
    
    return population


def genetic_algorithm(circuit, generations=5, population_size=10):
    population = initialize_population(population_size)
    max_fitness_over_time = []
    avg_fitness_over_time = []
    for generation in range(1, 1+generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(chromosome, circuit) for chromosome in population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        avg_fitness_over_time.append(avg_fitness)
        # Selection
        # Simple selection strategy: sort by fitness and select top half
        population = [chromosome for _, chromosome in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        parents = population[:len(population)//2]

        population = grow_shrink(population)
        population = [mutate(chromosome) for chromosome in population]
        population = crossover(population)

        max_fitness = max(fitness_scores)
        max_fitness_over_time.append(max_fitness)
        print(f"Generation {generation}, Average Fitness: {avg_fitness}, Best Fitness: {max_fitness}")

    return population, max_fitness_over_time
    # Return the best individual and its score
    # best_index = np.argmax(fitness_scores)
    # return population[best_index], fitness_scores[best_index]

# TODO: this is hardcoded
def initialize_population(population_size):
    fac = zne.RichardsonFactory(scale_factors=[1, 3, 5])

    def i1():
        return [
            REMGene(p0 = 0.05, p1 = 0.05),
            DDDGene(rule = ddd.rules.xx),
        ]

    def i2():
        return [
            REMGene(p0 = 0.05, p1 = 0.05),
            ZNEGene(factory = fac, scale_noise = zne.scaling.fold_global, num_to_avg = 1)
        ]

    def i():
        opts = [i1, i2]
        return opts[random.randint(len(opts))]()
    
    return [
        i()
        for _ in range(population_size)
    ]


def print_pop(pop):
    for i,j in enumerate(pop):
        print(f'{i+1}. {j}')

if __name__ == '__main__':
    
    circuit = generate_ghz_circuit(N_QUBITS)
    # circuit = generate_w_circuit(N_QUBITS)
    # circuit = generate_rb_circuits(N_QUBITS, 10)[0] # TODO: some benchmarking circuit

    pop_size = 20
    generation_count = 10
    
    pop = initialize_population(pop_size)
        
    for generation in trange(generation_count,
        desc = 'genetic algorithm',
        unit = 'generation',
    ):
        # mutation
        pop = [mutate(i) for i in pop]
        
        # crossover
        pop = crossover(pop, times=1)
    
        # fitness testing
        fitnesses = np.zeros(len(pop))
        for i in trange(len(pop),
            desc = 'fitness calculation',
            unit = 'candidate',
        ):
            fitnesses[i] = evaluate_fitness(pop[i], circuit)

        # sort by fitnesses
        pop_fit = sorted(
            zip(pop, fitnesses),
            key = lambda v: -v[1],
        )

        # re-extract population
        pop = [
            i for (i, f) in pop_fit
        ]

        # logging
        print(f"\n\nGeneration {generation}, Average Fitness: {np.mean(fitnesses)}, Best Fitness: {np.max(fitnesses)}\n\n")
        # print_pop(pop)

        # repopulation
        half = len(pop) // 2
        rest = len(pop) - half # account for odd-length population sizes
        pop = pop[:half] + pop[:rest]
