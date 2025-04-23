# Config Module

The **config** module centralizes all configuration, constant definitions, command-line argument parsing, and logging setup for the PhosKinTime package. This module is designed to standardize configuration settings across the package, making it easy to adjust model parameters, file paths, logging behavior, and other key settings.

---

## Overview

The config module is composed of several submodules:

- **`constants.py`**  
  This file defines global constants used throughout the package. It includes:
  - **Model Settings:**  
    - `ODE_MODEL`: Selects the ODE model type (e.g., "distmod" for Distributive, "succmod" for Successive, "randmod" for Random).
    - `ESTIMATION_MODE`: Chooses the parameter estimation strategy ("sequential" or "normal").
    - A mapping (`model_names`) that converts internal model identifiers to human-readable names (stored in `model_type`).
  - **Time Points and Directories:**  
    - `TIME_POINTS`: A NumPy array of experimental time points (in minutes) for data fitting.
    - Directory paths such as `PROJECT_ROOT`, `OUT_DIR`, `DATA_DIR`, `INPUT_EXCEL`, and `LOG_DIR`.
  - **Plotting and Regularization Settings:**  
    - `COLOR_PALETTE`: A list of colors for plotting.
    - `USE_REGULARIZATION` and `LAMBDA_REG`: Settings to enable and control Tikhonov (L2) regularization during model fitting.
  - **Scoring Weights:**  
    Weights for the composite scoring function (`ALPHA_WEIGHT`, `BETA_WEIGHT`, `GAMMA_WEIGHT`, `DELTA_WEIGHT`, `MU_REG`) that combine error metrics such as RMSE, MAE, variance, MSE, and the L2 norm of parameters.

- **`config.py`**  
  This file handles command-line argument parsing and configuration extraction. It provides:
  - Custom parsers (e.g., `parse_bound_pair` and `parse_fix_value`) to validate and convert command-line inputs.
  - The `parse_args` function to define and parse the necessary CLI arguments (such as parameter bounds, fixed parameter values, bootstrapping iterations, and input file paths).
  - Utility functions like `ensure_output_directory` to create necessary directories.
  - The `extract_config` function, which aggregates all command-line arguments and constants into a unified configuration dictionary used by the rest of the package.
  - A composite scoring function (`score_fit`) that calculates a combined error score based on various error metrics and the L2 regularization penalty.

- **`logconf.py`**  
  This file sets up the logging system for the package. Key features include:
  - **Colored Console Logging:**  
    A custom `ColoredFormatter` formats log messages with colors (e.g., blue for INFO, red for ERROR) to improve readability in the console.
  - **Rotating File Logging:**  
    Log messages are also saved to files in the directory specified by `LOG_DIR`, with rotation settings (maximum bytes and backup count) to prevent log files from becoming too large.
  - **Setup Function:**  
    The `setup_logger` function initializes and returns a logger configured with both file and stream (console) handlers.

---

### Global Configuration

The constants defined in `constants.py` control major aspects of the modeling and estimation processes. For example, to switch between different kinetic models, update the `ODE_MODEL` value. Similarly, change `ESTIMATION_MODE` to "sequential" or "normal" depending on whether you want time-point-by-time-point estimation or a global fit over all time points.

### Logging

The logger configured in `logconf.py` is used to log progress, warnings, and errors. Both console and file logging are available. Log messages include time stamps, module names, log levels, and elapsed time, formatted with colors for easier debugging.

---

## Customization

- **Model & Estimation Settings:**  
  Adjust `ODE_MODEL` and `ESTIMATION_MODE` in `constants.py` to select different modeling strategies and parameter estimation routines.

- **Parameter Bounds and Fixed Values:**  
  These can be set via command-line arguments or within the configuration file. Custom parsers ensure values are correctly converted (e.g., converting "inf" to Python’s infinity).

- **Output Directories:**  
  Paths for data, results, and logs are automatically generated relative to `PROJECT_ROOT`. Modify these if necessary for your environment.

- **Logging Behavior:**  
  The logging format and rotation settings can be adjusted in `logconf.py` to suit your needs.

- **Scoring Function:**  
  The composite scoring function in `config.py` can be tuned by modifying the weights (`ALPHA_WEIGHT`, `BETA_WEIGHT`, etc.) to reflect the priorities of your analysis.