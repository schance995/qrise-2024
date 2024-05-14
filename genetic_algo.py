# population = list[chromosome]
# chromosome = list[gene] = ORDERED list of mitiq executables
# gene = parameterized error mitigator
# assume gene is executable, takes only circuit as parameter

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
from functools import partial
import gc
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import cirq
import numpy as np
from numpy import random as random
import matplotlib.pyplot as plt
from mitiq import rem, zne, ddd, Observable, PauliString, MeasurementResult, raw
from mitiq.benchmarks import generate_rb_circuits, generate_ghz_circuit, generate_w_circuit
from mitiq.benchmarks.randomized_clifford_t_circuit import generate_random_clifford_t_circuit
from mitiq.benchmarks.mirror_qv_circuits import generate_mirror_qv_circuit
import pandas as pd
from scipy.special import expit
from tqdm import tqdm, trange

# make sure text fits inside plots
plt.rcParams['figure.constrained_layout.use'] = True

# Do not recreate the simulator every time
SIMULATOR = cirq.DensityMatrixSimulator()

def get_serial_code():
    """gets a unique integer every time the algorithm is run for logging purposes"""
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    count_file_path = config_dir / "count.txt"
    if count_file_path.is_file():
        with open(count_file_path, "r") as count_file: # open file in read mode
            count = count_file.readline() # read data and try to increment by 1
        try:
            count = int(count) + 1
        except:
            count = 0
    else:
        count = 0
    # file is closed after leaving with

    with open(count_file_path, "w") as count_file: # open file again but in write mode
        count_file.write(str(count)) # write count to file
    # file is closed after leaving with

    return count

# hardcode genes
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

    def __init__(self, p0=0.05, p1=0.05, n_qubits=5):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
        self.n_qubits = n_qubits
    
    def executor(self, executable):
        icm = rem.generate_inverse_confusion_matrix(self.n_qubits, self.p0, self.p1)
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
    
    def __init__(self, factory, scale_noise, num_to_avg, obs):
        super().__init__()
        self.factory = factory
        self.scale_noise = scale_noise
        self.num_to_avg = num_to_avg
        self.obs = obs

    def executor(self, executable):
        return zne.mitigate_executor(
            executable,
            observable=self.obs,
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
        
    def __init__(self, rule, obs):
        super().__init__()
        self.rule = rule
        self.obs = obs

    def executor(self, executable):
        return ddd.mitigate_executor(
            executable,
            rule=self.rule,
            observable=obs
        )

# this is hardcoded
def execute(circuit: cirq.Circuit, noise_level: float = 0.002, p0: float = 0.05) -> MeasurementResult:
    """Execute a circuit with depolarizing noise of strength ``noise_level`` and readout errors ...
    """
    measurements = circuit[-1]
    circuit =  circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))
    circuit.append(cirq.bit_flip(p0).on_each(circuit.all_qubits()))
    circuit.append(measurements)

    result = SIMULATOR.run(circuit, repetitions=1000)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


def ideal(circuit, obs):
    return raw.execute(circuit, partial(execute, noise_level=0, p0=0), obs)

def noisy(circuit, obs):
    return raw.execute(circuit, execute, obs)

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


def compute_mitigation_ratio(ideal_measurement, noisy_measurement, mitigated_measurement):
    if (distance_mitigated := abs(mitigated_measurement - ideal_measurement)) == 0:
        return 0
    if (distance_noisy     := abs(noisy_measurement     - ideal_measurement)) == 0:
        distance_noisy = 1e-8

    return distance_mitigated / distance_noisy


def compute_fitness(ideal_measurement = None, noisy_measurement = None, mitigated_measurement = None, ratio = None):
    if ratio is None:
        ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, mitigated_measurement)

    if ratio == 0:
        return 1

    return expit(-np.log(ratio)) # bound fitness between [0, 1]


def evaluate_fitness(chromosome, circuit, obs):
    """
    Evaluates the mitigation performance of 'chromosome' on 'circuit'
    - fitness = relative gain in mitigation
    - higher fitness = the difference between noisy and ideal > mitigated and ideal
    - ideal noise is as far away as possible
    - maximize negative sigmoid log ratio of differences
    """
    ideal_measurement = ideal(circuit, obs)
    noisy_measurement = noisy(circuit, obs)
    try:
        mitigated_measurement = mitigated(chromosome, circuit, execute)
    except zne.inference.ExtrapolationError:
        # really really bad if it doesn't even work
        # this means we can't log the worst fitness configurations
        return 0 # -1e8

    return compute_fitness(ideal_measurement, noisy_measurement, mitigated_measurement)


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

# this is hardcoded
def initialize_population(population_size, n_qubits, obs):
    """
    Initializes the population to 2 chromosome configurations: REM + DDD, or REM + ZNE. This is due to the limitations of the API and time constraints, but this can be expanded to cover more gene combinations if a more generalized error composition framework is developed.
    """
    def chain_1():
        return [
            REMGene(p0 = 0.05, p1 = 0.05, n_qubits = n_qubits),
            DDDGene(
                rule = random.choice(DDDGene.rules),
                obs = obs,
            ),
        ]

    def chain_2():
        return [
            REMGene(p0 = 0.05, p1 = 0.05, n_qubits = n_qubits),
            ZNEGene(
                factory = random.choice(ZNEGene.factories),
                scale_noise = random.choice(ZNEGene.scale_noises),
                num_to_avg = 1,
                obs = obs,
            ),
        ]

    def make_chain():
        known_working = [chain_1, chain_2]
        return known_working[random.randint(len(known_working))]()
    
    return [
        make_chain()
        for _ in range(population_size)
    ]

def genetic_algorithm(pop_size, generation_count, circuit, n_qubits, obs):
    '''
    Optimize a population of 'pop size' on 'circuit' for 'generation_count' generations.
    '''
    pop = initialize_population(pop_size, n_qubits, obs)
    max_fitness_over_time = []
    mean_fitness_over_time = []
    med_fitness_over_time = []
    max_indivs_over_time = []
    med_indivs_over_time = []
    best_max_fitness_so_far = float('-inf')
    best_mean_fitness_so_far = float('-inf')
    best_med_fitness_so_far = float('-inf')
    ctx = get_context('spawn')
    with ProcessPoolExecutor(mp_context=ctx, max_workers=20) as executor:
        for generation in (pbar := trange(generation_count,
            desc = 'genetic algorithm',
            unit = 'generation',
        )):
            # mutation
            pop = [mutate(i) for i in pop]

            # crossover
            pop = crossover(pop, times=1)
        
            # fitness testing. Multiprocessing speeds this up
            futures = [executor.submit(evaluate_fitness, indiv, circuit, obs) for indiv in pop]
            fbar = trange(len(futures), leave=False)
            def get_res(f, indiv, circuit):
                fbar.update()
                # execute the exceptions synchronously if unpickleable:
                # AttributeError: Can't pickle local object 'PolyFactory.extrapolate.<locals>.zne_curve'
                try:
                    result = f.result()
                except Exception as e:  # assume that errors are related to multiprocessing
                    # print(e, file=sys.stderr)
                    result = evaluate_fitness(indiv, circuit, obs)
                return result

            fitnesses = [get_res(future, indiv, circuit) for future, indiv in zip(futures, pop)]

            # sort by fitnesses
            pop_fit = sorted(
                zip(pop, fitnesses),
                key = lambda v: -v[1],
            )

            # logging
            max_indiv, max_fit = pop_fit[0]
            # array is sorted, use constant-time lookups
            med_indiv, med_fit = pop_fit[len(fitnesses)//2] 
            mean_fit = sum(p[1] for p in pop_fit) / len(pop_fit)

            best_max_fitness_so_far = max(best_max_fitness_so_far, max_fit)
            best_med_fitness_so_far = max(best_med_fitness_so_far, med_fit)
            best_mean_fitness_so_far = max(best_mean_fitness_so_far, mean_fit)

            max_indivs_over_time.append(max_indiv)
            max_fitness_over_time.append(max_fit)
            mean_fitness_over_time.append(mean_fit)
            med_fitness_over_time.append(med_fit)
            med_indivs_over_time.append(med_indiv)

            # re-extract population
            pop = [
                i for (i, f) in pop_fit
            ]

            # logging
            pbar.set_postfix_str(f"Best Mean Fit: {best_mean_fitness_so_far:.3f}, Best Med Fit: {best_med_fitness_so_far:.3f}, Best Max Fit: {best_max_fitness_so_far:.3f}")

            # print_pop(pop)

            # return the final population and metrics once done
            if generation + 1 >= generation_count:
                results = {
                    "max_fitness": max_fitness_over_time,
                    "mean_fitness": mean_fitness_over_time,
                    "med_fitness": med_fitness_over_time,
                    "max_indivs": max_indivs_over_time,
                    "med_indivs": med_indivs_over_time,
                }
                return pop, results

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

def make_plot(max_fits, mean_fits, med_fits, title, serial_code):
    fig, ax = plt.subplots()
    ticks = list(range(1, 1 + len(max_fits)))
    ax.plot(ticks, max_fits, label='Max')
    ax.plot(ticks, mean_fits, label='Mean')
    ax.plot(ticks, med_fits, label='Median')
    ax.set(xlabel='Generation', ylabel='Fitness', title=title)
    ax.set_xticks(ticks)
    ax.legend()
    ax.grid()
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{title} (run {serial_code}).png')
    plt.close()
    gc.collect()

def benchmark_results(fittest_chromosome, circuit, n_qubits, obs):
    print("\n========= BENCHMARK RESULTS ========")


    ideal_measurement = ideal(circuit, obs)
    noisy_measurement = noisy(circuit, obs)
    print("Ideal value:", "{:.5f}".format(ideal_measurement.real))
    print("Noisy value:", "{:.5f}".format(noisy_measurement.real))

    icm = rem.generate_inverse_confusion_matrix(n_qubits, 0.05, 0.05) # arbitrary config
    rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)

    rem_result = obs.expectation(circuit, rem_executor)
    print("Mitigated value obtained with REM:", "{:.5f}".format(rem_result.real))

    zne_result = zne.execute_with_zne(circuit, execute, obs) # default params
    print("Mitigated value obtained with ZNE:", "{:.5f}".format(zne_result.real))

    ddd_result = ddd.execute_with_ddd(circuit, execute, obs, rule = ddd.rules.xx) # default params
    print("Mitigated value obtained with DDD:", "{:.5f}".format(ddd_result.real))

    optim_result = mitigated(fittest_chromosome, circuit, execute)
    print("Optim mitigated value:", "{:.5f}".format(optim_result.real))

    # compute mitigation ratios of each
    rem_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, rem_result)
    zne_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, zne_result)
    ddd_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, ddd_result)
    dna_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, optim_result)
    ratios = {
        'REM': rem_ratio,
        'ZNE': zne_ratio,
        'DDD': ddd_ratio,
        'DNA': dna_ratio,
    }
    print('REM ratio: {:.5f}'.format(rem_ratio))
    print('ZNE ratio: {:.5f}'.format(zne_ratio))
    print('DDD ratio: {:.5f}'.format(ddd_ratio))
    print('DNA ratio: {:.5f}'.format(dna_ratio))

    rem_fitness = compute_fitness(ratio=rem_ratio)
    zne_fitness = compute_fitness(ratio=zne_ratio)
    ddd_fitness = compute_fitness(ratio=ddd_ratio)
    dna_fitness = compute_fitness(ratio=dna_ratio)
    fitnesses = {
        'REM': rem_fitness,
        'ZNE': zne_fitness,
        'DDD': ddd_fitness,
        'DNA': dna_fitness,
    }
    print('REM fitness: {:.5f}'.format(rem_fitness))
    print('ZNE fitness: {:.5f}'.format(zne_fitness))
    print('DDD fitness: {:.5f}'.format(ddd_fitness))
    print('DNA fitness: {:.5f}'.format(dna_fitness))
    return fitnesses, ratios


def generate_circuits(n_qubits):
    return [
        generate_ghz_circuit(n_qubits),
        generate_w_circuit(n_qubits),
        generate_random_clifford_t_circuit(
            num_qubits=n_qubits,
            num_oneq_cliffords=n_qubits,
            num_twoq_cliffords=n_qubits,
            num_t_gates=n_qubits,
        ),
        # what do mirror circuits return?
        # ValueError: probabilities are not non-negative
        # generate_mirror_qv_circuit(
        #     num_qubits=n_qubits,
        #     depth=n_qubits,
        # ),
    ]

def run_experiment(circuit, circuit_names, n_qubits, obs, serial_code):
    benchmark_fitnesses = [] # {circuit_name: [] for circuit_name in circuit_names}
    benchmark_ratios = [] # {circuit_name: [] for circuit_name in circuit_names}
    with open(f"output_{serial_code}.txt", "a") as f:
        with redirect_stdout(f): # this is a temporary workaround for logging because we couldn't get the logging module working. This current redirects ALL stdout to the file so it could capture random unintended outputs.
            print(f"\n\n\n############### EXPERIMENT {serial_code} ({time.asctime()}) ############")

            for circuit, circuit_name in zip(circuits, circuit_names):
                title = f'{circuit_name} with {n_qubits} qubits and seed {seed}'
                print(f'\n\nCIRCUIT: {title}')
                print(circuit)  # SVG cirq drawings are unreliable
                final_pop, results = genetic_algorithm(pop_size, generation_count, circuit, n_qubits, obs)
                max_indivs = results['max_indivs']
                med_indivs = results['med_indivs']
                max_fits = results['max_fitness']
                mean_fits = results['mean_fitness']
                med_fits = results['med_fitness']
                print('Final pop')
                print_pop(final_pop)
                print('Max pop')
                print_pop(max_indivs)
                print('Med pop')
                print_pop(med_indivs)
                make_plot(max_fits, mean_fits, med_fits, title, serial_code)

                # use the best max individual
                best_max_indiv = max_indivs[np.argmax(max_fits)]
                print('Best max individual')
                print(best_max_indiv)
                fitnesses, ratios = benchmark_results(best_max_indiv, circuit, n_qubits, obs)
                # convert dictionaries to dataframes
                benchmark_fitnesses.append(pd.DataFrame(fitnesses, index=[circuit_name]))
                benchmark_ratios.append(pd.DataFrame(ratios, index=[circuit_name]))
    return pd.concat(benchmark_fitnesses), pd.concat(benchmark_ratios)


def plot_benchmarks(benchmark_fitnesses, benchmark_ratios, n_qubits, serial_code):

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    fitness_df = pd.concat(benchmark_fitnesses)
    mean_fitness_df = fitness_df.groupby(fitness_df.index).mean()
    # std_fitness_df = fitness_df.groupby(fitness_df.index).std()
    ax = mean_fitness_df.plot.bar(rot=0) # , yerr=std_fitness_df)
    title = f'Mitigation fitness with {n_qubits} qubits (run {serial_code})'
    ax.set(xlabel='Mitigation', ylabel='Fitness', title=title)
    plt.savefig(plots_dir / f'{title}.png')
    plt.close()
    gc.collect() # https://github.com/matplotlib/matplotlib/issues/27713

    ratio_df = pd.concat(benchmark_ratios)
    mean_ratio_df = ratio_df.groupby(ratio_df.index).mean()
    # std_ratio_df = ratio_df.groupby(ratio_df.index).std()
    ax = mean_ratio_df.plot.bar(rot=0) # , yerr= std_ratio_df) # only include 1 standard error
    title = f'Mitigation ratio with {n_qubits} qubits (run {serial_code})'
    ax.set(xlabel='Mitigation', ylabel='Ratio', title=title)
    plt.savefig(plots_dir / f'{title}.png')
    plt.close()
    gc.collect()

if __name__ == '__main__':

    serial_code = get_serial_code()
    pop_size = 40  # TODO: set to 40 when ready
    generation_count = 10  # TODO: set to 10 when ready
    max_qubits = 7
    n_seeds = 3  # TODO: set to 3 when ready

    for n_qubits in range(6, max_qubits+1):
        benchmark_fitnesses = []
        benchmark_ratios = []
        for seed in range(n_seeds):
            np.random.seed(seed) # global random seed is probably not respected
            obs = Observable(PauliString("Z" * n_qubits))
            circuits = generate_circuits(n_qubits)
            circuit_names = [
                'GHZ',
                'W-state',
                'Random Clifford T',
            ]
            these_benchmark_fitnesses, these_benchmark_ratios = run_experiment(circuits, circuit_names, n_qubits, obs, serial_code)
            benchmark_fitnesses.append(these_benchmark_fitnesses)
            benchmark_ratios.append(these_benchmark_ratios)

        plot_benchmarks(benchmark_fitnesses, benchmark_ratios, n_qubits, serial_code)

