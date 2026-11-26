# API Reference

## Data Standardization & Cleanup

::: processing.cleanup 

## Optimization Results Mapping

::: processing.map

## ODE Modelling & Parameter Estimation

### Configuration
 
::: config.cli
::: config.config
::: config.constants
::: config.logconf

### Core Functions

::: protwise.paramest.normest
::: protwise.paramest.toggle

### Weights for Curve Fitting

::: protwise.models.weights

### Parameter Estimation

::: protwise.paramest.core

### Confidence Intervals using Linearization

::: protwise.paramest.identifiability.ci

### Knockout Analysis

::: protwise.knockout.helper

### Perturbation & Parameter Sensitivity Analysis

::: protwise.sensitivity.analysis

### Model Diagram

::: protwise.models.diagram.helpers

### Protein Wise Model Types

::: protwise.models.distmod
::: protwise.models.randmod
::: protwise.models.succmod

### Steady-State Calculation

::: protwise.steady.initdist
::: protwise.steady.initrand
::: protwise.steady.initsucc

### Plotting

::: protwise.plotting.plotting

### Utility Functions

::: common.utils.display
::: common.utils.tables
::: common.utils.latexit 

## Global ODE Model

### Core Data Structures & Topology

::: networkmodel.network
::: networkmodel.BuildMatrix
::: networkmodel.params

### Configuration & Data Loading

::: networkmodel.config
::: networkmodel.io

### Physics Kernels (JIT)

::: networkmodel.models
::: networkmodel.TimeBucket

### Numerical Integration & Solvers

::: networkmodel.simulate
::: networkmodel.backend
::: networkmodel.SteadyStateAnalysis
::: networkmodel.InitialConditions

### Optimization & Loss Functions

::: networkmodel.OptimizationProblem
::: networkmodel.LossFunction
::: networkmodel.BayesianInference
::: networkmodel.PosteriorObjective
::: networkmodel.PosteriorWorker
::: networkmodel.ProfileWorker
::: networkmodel.runner
::: networkmodel.scan

### Analysis & Visualization

::: networkmodel.sensitivity
::: networkmodel.export
::: networkmodel.dashboard_app
::: networkmodel.dashboard_bundle
::: networkmodel.mode_outputs

### Utilities

::: networkmodel.utils
::: networkmodel.cache