### Overview
This is a writeup of the different hyperparmeters we can tune for the 3 QEM techniques.

### REM
- p0: probability of 0 flipping to 1
  - Possible values: a float in [0, 1]
- p1: probability of 1 flippping to 0
  - Possible values: a float in [0, 1]

### ZNE
- factory: Factory object determining the zero-noise extrapolation method.
  - Possible values: 
    - zne.RichardsonFactory
    - zne.PolyFactory
    - zne.PolyExpFactory
    - zne.AdaExpFactory
    - zne.LinearFactory
    - zne.ExpFactory
  - Also note that each factory has tunable hyperparameters
    - scale_factors: Sequence of noise scale factors at which expectation values should be measured.
    - shot_list: Optional sequence of integers corresponding to the number of samples taken for each expectation value. If this argument is explicitly passed to the factory, it must have the same length of scale_factors and the executor function must accept "shots" as a valid keyword argument.
- scale_noise: Function for scaling the noise of a quantum circuit.
  - Possible values: 
    - zne.scaling.fold_global
    - zne.scaling.fold_all
    - zne.scaling.fold_gates_from_left
    - zne.scaling.fold_gates_from_right
    - zne.scaling.fold_gates_at_random
    - zne.scaling.insert_id_layers
    - zne.scaling.layer_folding
- num_to_avg: number of times expectation values are computed by the executor after each call to scale_noise, then averaged.
  - Possible value: any positive integer

### DDD
- rule: determines what DDD sequence should be applied in a given slack window.
  - Possible values: ddd.rules.xx, ddd.rules.yy, ddd.rules.xyxy