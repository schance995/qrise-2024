# population = list[chromosome]
# chromosome = list[gene] = list of mitiq executables
# gene = parameterized error mitigator
# assume gene is executable, takes only circuit as parameter

from abc import ABC, abstractmethod
from functools import partial
from mitiq.benchmarks import generate_rb_circuits
from mitiq import rem, zne, Observable, PauliString, MeasurementResult, raw
import cirq
import numpy as np

N_QUBITS = 2
OBS = Observable(PauliString("Z"*N_QUBITS)) # THIS IS IN THE WRONG PLACE

# TODO: for now I'm just going to hardcode the genes
class BaseGene(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def executor(self, executable):
        pass
    
class REMGene(BaseGene):
    def __init__(self, p0=0.05, p1=0.05):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
    
    def executor(self, executable):
        icm = rem.generate_inverse_confusion_matrix(N_QUBITS, self.p0, self.p1)

        return rem.mitigate_executor(executable, inverse_confusion_matrix=icm)

class ZNEGene(BaseGene):
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

    # result = chain_executor(circuit)
    result = OBS.expectation(circuit, chain_executor)

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
    fitness = 1 - abs(mitigated_measurement - ideal_measurement) / abs(noisy_measurement - ideal_measurement)
    return fitness


def mutate(chromosome, p=0.5):
    return chromosome
    # if np.random.uniform(0,1) < p: # some p
    #     return new(chromosome) # or change its parameter
    # return chromosome


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

# TODO: this is hardcoded
def initialize_population(population_size):
    fac = zne.RichardsonFactory(scale_factors=[1, 3, 5])

    # return [[REMGene(p0=0.05, p1=0.05), ZNEGene(factory=fac, scale_noise=zne.scaling.fold_global, num_to_avg=1)] for _ in range(population_size)]
    return [[REMGene(p0=0.05, p1=0.05)] for _ in range(population_size)]


circuit = generate_rb_circuits(2, 10)[0] # TODO: some benchmarking circuit
print(genetic_algorithm(circuit))
