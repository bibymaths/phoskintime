# Steady Module 

The **steady** module provides functions to compute the steady‐state initial conditions required for simulating the ODE-based phosphorylation models within the PhosKinTime package. These functions solve the steady-state equations—where all derivatives are zero—using optimization routines. The steady-state conditions are critical for ensuring that the model simulations start from biologically meaningful states.

## Overview

The module includes several implementations tailored to different modeling approaches:

- **`initdist.py` (Distributive Model):**  
  Calculates the steady-state initial conditions for models where phosphorylation occurs in a distributive manner. This function assumes that phosphorylation events are independent and solves for the states of the system by minimizing a dummy objective function subject to the steady-state constraints.

- **`initrand.py` (Random Model):**  
  Provides an implementation for the random model. It accounts for all possible combinations of phosphorylation states by using binary representations and generates the corresponding steady-state conditions using a constraint-based optimization approach.

- **`initsucc.py` (Successive Model):**  
  Computes steady-state initial conditions for models with a successive phosphorylation mechanism, where phosphorylation occurs sequentially. It uses a similar optimization approach to solve the steady-state equations.

Each of these files defines an `initial_condition(num_psites: int) -> list` function that:
- Sets up the steady-state equations based on fixed parameter values (e.g., A, B, C, D, S_rates, and D_rates).
- Provides a reasonable initial guess.
- Uses `scipy.optimize.minimize` (with the SLSQP method) to find a solution where all time derivatives are zero.
- Returns the computed steady-state values as a list.

## Key Features

- **Model-Specific Steady-State Calculation:**  
  Each function is tailored to the specifics of the underlying model (distributive, random, or successive), ensuring that the appropriate steady-state conditions are determined.

- **Constraint-Based Optimization:**  
  The steady-state is calculated by solving a set of equality constraints (i.e., setting the derivatives to zero) using the SLSQP algorithm, ensuring that the returned initial conditions satisfy the model dynamics.

- **Logging:**  
  Logging is integrated via the package’s logging configuration (`config/logconf.py`), allowing users to monitor the progress and success of the steady-state calculation.

## Usage

To obtain the steady-state initial conditions for a given number of phosphorylation sites, simply import and call the appropriate function based on your model type. For example:

```python
# For the distributive model:
from steady.initdist import initial_condition

num_psites = 3  # Example: 3 phosphorylation sites
init_cond = initial_condition(num_psites)
print("Steady-state initial conditions (distributive):", init_cond)

# For the random model:
from steady.initrand import initial_condition

init_cond_rand = initial_condition(num_psites)
print("Steady-state initial conditions (random):", init_cond_rand)

# For the successive model:
from steady.initsucc import initial_condition

init_cond_succ = initial_condition(num_psites)
print("Steady-state initial conditions (successive):", init_cond_succ)
```

These initial conditions are then used as input for ODE solvers within the package to simulate the phosphorylation dynamics.

## Conclusion

The **steady** module is a fundamental part of the PhosKinTime package, ensuring that simulations begin from a biologically realistic and mathematically consistent state. By providing tailored steady-state solvers for various kinetic models, it supports robust and accurate ODE simulations for phosphorylation dynamics.

For more details or customization, refer to the source code in `initdist.py`, `initrand.py`, and `initsucc.py`.