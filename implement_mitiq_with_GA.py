import numpy as np
import cirq
from mitiq.benchmarks import generate_rb_circuits
from mitiq import MeasurementResult, Observable, PauliString, raw
from mitiq import rem
from mitiq import zne
import random


def execute(circuit: cirq.Circuit, noise_level: float = 0.002, p0: float = 0.05) -> MeasurementResult:
    """Execute a circuit with depolarizing noise of strength ``noise_level`` and readout errors ...
    """
    measurements = circuit[-1]
    circuit =  circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))
    circuit.append(cirq.bit_flip(p0).on_each(circuit.all_qubits()))
    circuit.append(measurements)

    simulator = cirq.DensityMatrixSimulator()

    result = simulator.run(circuit, repetitions=10000)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)



obs = Observable(PauliString("ZI"), PauliString("IZ"))
noisy = raw.execute(circuit, execute, obs)
from functools import partial

ideal = raw.execute(circuit, partial(execute, noise_level=0, p0=0), obs)
print("Unmitigated value:", "{:.5f}".format(noisy.real))

# Assume execute function is defined here

def initialize_population(size):
    # Initialize with random values for our parameters
    return [{"noise_scaling": random.choice(["lf", "gf", "ii"]),
             "p0": random.uniform(0.01, 0.1),
             "p1": random.uniform(0.01, 0.1),
             "noise_level": random.uniform(0.001, 0.01)} for _ in range(size)]

def evaluate_fitness(individual, circuit):
    p0, p1 = individual["p0"], individual["p1"]
    noise_level = individual["noise_level"]
    icm = rem.generate_inverse_confusion_matrix(2, p0, p1)
    rem_executor = rem.mitigate_executor(lambda circuit: execute(circuit, noise_level, p0), inverse_confusion_matrix=icm)
        
        # Initialize combined_executor with a default value (rem_executor in this case)
    combined_executor = rem_executor
    # Modify this part to apply ZNE based on the individual's parameters
    # Here's a simple placeholder for applying global folding
    if individual["noise_scaling"] == "gf":
        combined_executor = zne.mitigate_executor(rem_executor, scale_noise=zne.scaling.fold_global)
    # Add conditions for "lf" and "ii" with corresponding adjustments
    
    result = combined_executor(circuit)
    return result.real  # Assuming we want to maximize this value

def genetic_algorithm(circuit, generations=10, population_size=20):
    population = initialize_population(population_size)
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(individual, circuit) for individual in population]
        
        # Selection
        # Simple selection strategy: sort by fitness and select top half
        population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        parents = population[:len(population)//2]
        
        # TODO Crossover and Mutation
        # Implement crossover and mutation here please help idk what to do lol
        
        print(f"Generation {generation}, Best Score: {max(fitness_scores)}")
    
    # Return the best individual and its score
    best_index = np.argmax(fitness_scores)
    return population[best_index], fitness_scores[best_index]

# it says circuit does not have measurements :( 
circuit = generate_rb_circuits(2, 10)[0]

best_parameters, best_score = genetic_algorithm(circuit)
print(f"Best Parameters: {best_parameters}, Best Score: {best_score}")