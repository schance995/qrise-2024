# population = list[chromosome]
# chromosome = list[gene] = list of mitiq executables
# gene = parameterized error mitigator
# assume gene is executable, takes only circuit as parameter

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import cirq
import numpy as np
from numpy import random as random
from mitiq import rem, zne, ddd, Observable, PauliString, MeasurementResult, raw
from mitiq.benchmarks import generate_rb_circuits, generate_ghz_circuit, generate_w_circuit
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
    scale_noises = [
        zne.scaling.fold_global,
        zne.scaling.fold_all,
        zne.scaling.fold_gates_from_left,
        zne.scaling.fold_gates_from_right,
        zne.scaling.fold_gates_at_random,
        # zne.scaling.insert_id_layers,
        # zne.scaling.layer_folding,
    ]

    factories = [
        zne.inference.RichardsonFactory(scale_factors=[1, 3, 5]),
        zne.inference.LinearFactory(scale_factors=[1, 3]),
        zne.inference.PolyFactory(scale_factors=[1, 1.5, 2, 2.5, 3], order=2),
        zne.inference.ExpFactory(scale_factors=[1, 2, 3], asymptote=0.5),
        zne.inference.AdaExpFactory(steps=5, asymptote=0.5),
    ]

    @staticmethod
    def generate_factory():
        scale_factors = np.sort(
            random.uniform(
                low  = 1.0,
                high = 10.0,
                size = random.randint(2, 7),
            )
        )
        match random.randint(5):
            case 0: # Richardson
                return zne.inference.RichardsonFactory(
                    scale_factors = scale_factors
                )
            case 1: # Linear
                return zne.inference.LinearFactory(
                    scale_factors = scale_factors
                )
            case 2: # Poly
                return zne.inference.PolyFactory(
                    scale_factors = scale_factors,
                    order = random.randint(len(scale_factors))
                )
            case 3: # Exp
                return zne.inference.ExpFactory(
                    scale_factors = scale_factors,
                    asymptote = random.uniform(
                        low  = -10.0,
                        high = +10.0,
                    )
                )
            case 4: # AdaExp
                match random.randint(2):
                    case 0:
                        # without asymptote
                        return zne.inference.AdaExpFactory(
                            steps = random.randint(4, 15),
                        )
                    case 1:
                        # with asymptote
                        return zne.inference.AdaExpFactory(
                            steps = random.randint(3, 15),
                            asymptote = random.uniform(
                                low  = -10.0,
                                high = +10.0,
                            )
                        )
    
    def __str__(self):
        factory_name = self.factory.__class__.__name__
        parameters = ''
        match self.factory:
            case zne.inference.RichardsonFactory(_scale_factors=scale_factors):
                parameters = f'{scale_factors}'
            case zne.inference.LinearFactory(_scale_factors=scale_factors):
                parameters = f'{scale_factors}'
            case zne.inference.PolyFactory(_scale_factors=scale_factors, _options=options):
                parameters = f'{scale_factors}, {options["order"]}'
            case zne.inference.ExpFactory(_scale_factors=scale_factors, _options=options):
                parameters = f'{scale_factors}, {options["asymptote"]}'
            case zne.inference.AdaExpFactory(_steps=steps, asymptote=asymptote):
                parameters = f'{steps}, {asymptote}'
        return f'zne({factory_name}({parameters}), {self.scale_noise.__name__})'
    
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
            num_to_average=self.num_to_avg,
        )
    
class DDDGene(BaseGene):
    rules = [
        ddd.rules.xx,
        ddd.rules.xyxy,
        ddd.rules.yy,
    ]
    
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
    ideal_measurement = ideal(circuit)
    noisy_measurement = noisy(circuit)
    mitigated_measurement = mitigated(chromosome, circuit, execute)
    # fitness = relative gain in mitigation
    # higher fitness = the difference between noisy and ideal > mitigated and ideal
    # ideal noise is as far away as possible
    # maximize negative tanh log ratio of differences

    if (distance_mitigated := abs(mitigated_measurement - ideal_measurement)) == 0:
        fitness = 1
    
    # fix for divide by zero and log 0 in fitness :(
    else:
        if (distance_noisy     := abs(noisy_measurement     - ideal_measurement)) == 0:
            distance_noisy = 1e-8

        relative_distance = distance_mitigated / distance_noisy
        fitness = np.tanh(-np.log(relative_distance))

    return fitness


def mutate(chromosome):
    i = random.randint(len(chromosome))
    c = chromosome[i]
    # if no change happened, consider it as a no-op
    match c:
        case REMGene(p0=p0, p1=p1):
            match random.randint(3):
                case 0:
                    pass
                case 1:
                    c.p0 += 0.01 * random.randn()
                case 2:
                    c.p1 += 0.01 * random.randn()
            c.p0 = np.clip(c.p0, 0, 1)
            c.p1 = np.clip(c.p1, 0, 1)
        case ZNEGene(scale_noise=scale_noise):
            match random.randint(3):
                case 0:
                    c.scale_noise = random.choice(ZNEGene.scale_noises)
                case 1:
                    c.factory = ZNEGene.generate_factory()
                case 2:
                    match c.factory:
                        case zne.inference.RichardsonFactory(_scale_factors=scale_factors):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.LinearFactory(_scale_factors=scale_factors):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.PolyFactory(_scale_factors=scale_factors, _order=order):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.ExpFactory(_scale_factors=scale_factors, _asymptote=asymptote):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.AdaExpFactory(_steps=steps, _asymptote=asymptote):
                            c.factory.steps = c.factory.steps + random.randint(3) - 1
        case DDDGene(rule=rule):
            c.rule = random.choice(DDDGene.rules)
    return chromosome

def crossover(population, times=None):
    if times is None:
        times = len(population) // 2

    # individual crossover
    def cross(x, y, i=0):
        ix = random.randint(len(x))
        iy = random.randint(len(y))
        # right now, we can only swap matching gene types
        if type(x[ix]) == type(y[iy]):
            t     = x[ix]
            x[ix] = y[iy]
            y[iy] = t
        else:
            # failed, try again 10 times
            if i < 10:
                cross(x, y, i+1)
    
    # do an arbitrary number of crossover iterations
    for _ in range(times):
        ix, iy = random.randint(len(population), size=2)
        cross(population[ix], population[iy], 0)
    
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
    def chain_1():
        return [
            REMGene(p0 = 0.05, p1 = 0.05),
            DDDGene(
                rule = random.choice(DDDGene.rules)
            ),
        ]

    def chain_2():
        return [
            REMGene(p0 = 0.05, p1 = 0.05),
            ZNEGene(
                factory = random.choice(ZNEGene.factories),
                scale_noise = random.choice(ZNEGene.scale_noises),
                num_to_avg = 1,
            )
        ]

    def make_chain():
        known_working = [chain_1, chain_2]
        return known_working[random.randint(len(known_working))]()
    
    return [
        make_chain()
        for _ in range(population_size)
    ]


def genetic_algorithm(pop_size, generation_count, circuit):
    '''
    Optimize a population of 'pop size' on 'circuit' for 'generation_count' generations.
    '''
    pop = initialize_population(pop_size)

    for generation in (pbar := trange(generation_count,
        desc = 'genetic algorithm',
        unit = 'generation',
    )):
        # mutation
        pop = [mutate(i) for i in pop]
        
        # crossover
        pop = crossover(pop, times=1)
    
        # fitness testing
        fitnesses = np.zeros(len(pop))
        for i in trange(len(pop),
            desc = 'fitness calculation',
            unit = 'candidate',
            leave = False,
        ):
            try:
                fitnesses[i] = evaluate_fitness(pop[i], circuit)
            except zne.inference.ExtrapolationError:
                # really really bad if it doesn't even work
                fitnesses[i] = -1e8

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
        pbar.set_postfix_str(f"Median Fitness: {np.median(fitnesses):.3f}, Best Fitness: {np.max(fitnesses):.3f}")
        # print(f"\n\nGeneration {generation}, Median Fitness: {np.median(fitnesses)}, Best Fitness: {np.max(fitnesses)}\n\n")
        # print_pop(pop)

        if generation >= generation_count + 1:
            return pop

        # repopulation
        half = len(pop) // 2
        rest = len(pop) - half # account for odd-length population sizes
        pop = [
            [
                deepcopy(j)
                for j in i
            ]
            for i in pop[:half] + pop[:rest]
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

    final_pop = genetic_algorithm(pop_size, generation_count, circuit)
    print_pop(final_pop)
