# Config Helper Module

The **config/helper** module provides a set of utility functions that support configuration and parameter management across the PhosKinTime package. These helper functions are designed to generate descriptive parameter names, state labels, and parameter bounds for different kinetic models. They also include utilities for formatting file paths as clickable links in supported terminals.

## Key Functions

### Parameter Name Generators

- **`get_param_names_rand(num_psites: int) -> list`**  
  Generates parameter names for the **random model**.  
  - **Format:**  
    `['A', 'B', 'C', 'D']` (base parameters)  
    `+ ['S1', 'S2', ..., 'S<num_psites>']` (phosphorylation parameters)  
    `+ [<dephosphorylation parameters for all combinations>]`  
  - **Usage:**  
    Used when the ODE model is set to `"randmod"`.

- **`get_param_names_ds(num_psites: int) -> list`**  
  Generates parameter names for **distributive or successive models**.  
  - **Format:**  
    `['A', 'B', 'C', 'D'] + ['S1', 'S2', ..., 'S<num_psites>'] + ['D1', 'D2', ..., 'D<num_psites>']`  
  - **Usage:**  
    Used when the ODE model is not `"randmod"`.

### Label Generators

- **`generate_labels_rand(num_psites: int) -> list`**  
  Generates state labels for the **random model** based on the number of phosphorylation sites.  
  - **Example (num_psites = 2):**  
    `["R", "P", "P1", "P2", "P12"]`

- **`generate_labels_ds(num_psites: int) -> list`**  
  Generates state labels for **distributive or successive models**.  
  - **Example (num_psites = 2):**  
    `["R", "P", "P1", "P2"]`

### Other Utilities

- **`location(path: str, label: str = None) -> str`**  
  Returns a clickable hyperlink string using ANSI escape sequences for terminals that support it.  
  - **Usage:**  
    Useful for printing file paths as clickable links in the console.

- **`get_number_of_params_rand(num_psites: int) -> int`**  
  Calculates the total number of parameters required for the random model, taking into account:
  - Base parameters (`A`, `B`, `C`, `D`)
  - Phosphorylation parameters (one per site)
  - Dephosphorylation parameters (for each non-empty combination of sites)

- **`get_bounds_rand(num_psites: int, ub: float = 0, lower: float = 0) -> list`**  
  Generates a list of bounds for the ODE parameters for the random model.
  - **Format:**  
    A list of `[lower, ub]` pairs, one for each parameter (base, phosphorylation, and dephosphorylation parameters).

## How It Fits in the Package

These helper functions are used throughout the PhosKinTime package to:

- **Configure Parameter Estimation:**  
  Provide meaningful names and bounds for kinetic parameters depending on the chosen model type.

- **Generate Descriptive Labels:**  
  Create state labels that are used in plotting and reporting, ensuring that outputs are both informative and human-readable.

- **File and Terminal Utilities:**  
  Format file paths and hyperlinks to improve the clarity of console output and debugging.

## Example Usage

Below is an example of how these functions might be used within the package:

```python
from config.helper import get_param_names_rand, generate_labels_rand, location, get_number_of_params_rand, get_bounds_rand

num_psites = 3
# For the random model:
param_names = get_param_names_rand(num_psites)
state_labels = generate_labels_rand(num_psites)
print("Parameter Names:", param_names)
print("State Labels:", state_labels)

# Get total number of parameters required for the random model:
total_params = get_number_of_params_rand(num_psites)
print("Total Number of Parameters:", total_params)

# Generate bounds for parameters (example: lower bound 0, upper bound 100)
bounds = get_bounds_rand(num_psites, ub=100, lower=0)
print("Parameter Bounds:", bounds)

# Format a file location as a clickable link (if terminal supports it)
file_link = location("/path/to/results.xlsx", label="Results File")
print("File Link:", file_link)
```

## Conclusion

The **config/helper** module is a central component of the PhosKinTime package that standardizes and simplifies configuration tasks related to parameter naming, labeling, and bounds generation. Its utility functions ensure that model parameters are clearly defined and consistent throughout the package, enhancing both usability and maintainability.

---