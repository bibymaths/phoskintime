# Sensitivity Analysis

The Sensitivity module provides functionality for performing sensitivity analysis on the ODE-based phosphorylation models in the PhosKinTime package. Using the Morris method from SALib, this module evaluates the influence of each model parameter on the output of the system, thereby helping to identify the most critical parameters and potential nonlinear interactions.

## Overview

This module (primarily implemented in `analysis.py`) defines functions that:
- **Define the Sensitivity Problem:**  
  Two functions (`define_sensitivity_problem_rand` and `define_sensitivity_problem_ds`) generate the problem definition (number of variables, parameter names, and bounds) required for the Morris sensitivity analysis. The choice depends on whether the model is a random model (`randmod`) or a distributive/successive model.
  
- **Run Sensitivity Analysis:**  
  The `sensitivity_analysis` function:
  - Generates parameter samples using the Morris method.
  - Simulates the ODE system (via the package's `solve_ode` function) for each parameter set.
  - Computes a response metric (e.g., the sum of the phosphorylated states at the final time point).
  - Analyzes the sensitivity indices using SALib's `analyze` function.
  - Generates a suite of plots (bar plots, scatter, radial, CDF, and pie charts) to visually summarize the sensitivity of each parameter.
 
--- 

## Features

- **Model-Specific Sensitivity Problem Definition:**  
  Generates problem definitions tailored to either the random model or the distributive/successive models, automatically adjusting the number of parameters and their bounds.

- **Comprehensive Sensitivity Analysis:**  
  Uses the Morris method to provide:
  - **Mu\* (Mean Absolute Effect):** Indicates overall parameter importance.
  - **Sigma (Standard Deviation):** Reflects the degree of nonlinearity or interaction in parameter effects.

- **Automated Plotting:**  
  The module creates and saves multiple plots:
  - Bar plots for mu\* and sigma.
  - Scatter plot of mu\* versus sigma.
  - A radial (spider) plot summarizing sensitivity metrics.
  - Cumulative Distribution Function (CDF) plot of sensitivity indices.
  - Pie chart showing the proportional sensitivity contribution of each parameter.
 
--- 

## Usage

The main function in this module is `sensitivity_analysis`, which can be called with the following parameters:

- `time_points`: A NumPy array of experimental time points.
- `num_psites`: Number of phosphorylation sites.
- `init_cond`: The initial conditions for the ODE simulation.
- `gene`: The gene (or protein) identifier for which sensitivity is analyzed.