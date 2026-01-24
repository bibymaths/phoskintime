# API Reference

## Data Standardization & Cleanup

::: processing.cleanup 

## Optimization Results Mapping

::: processing.map

## Kinase-Phosphorylation Optimization

### Evolutionary Algorithms

::: kinopt.evol.config.constants
::: kinopt.evol.config.logconf
::: kinopt.evol.exporter.plotout
::: kinopt.evol.exporter.sheetutils
::: kinopt.evol.objfn.minfndiffevo
::: kinopt.evol.objfn.minfnnsgaii
::: kinopt.evol.opt.optrun
::: kinopt.evol.optcon.construct
::: kinopt.evol.utils.iodata
::: kinopt.evol.utils.params

### Gradient-Based Algorithms

::: kinopt.local.config.constants
::: kinopt.local.config.logconf
::: kinopt.local.exporter.plotout
::: kinopt.local.exporter.sheetutils
::: kinopt.local.objfn.minfn
::: kinopt.local.opt.optrun
::: kinopt.local.optcon.construct
::: kinopt.local.utils.iodata
::: kinopt.local.utils.params

## Fitting Analysis & Feasibility

::: kinopt.fitanalysis.helpers.postfit
::: kinopt.optimality.KKT

## TF-mRNA Optimization

### Evolutionary Algorithms

::: tfopt.evol.config.constants
::: tfopt.evol.config.logconf
::: tfopt.evol.exporter.plotout
::: tfopt.evol.exporter.sheetutils
::: tfopt.evol.objfn.minfn
::: tfopt.evol.opt.optrun
::: tfopt.evol.optcon.construct
::: tfopt.evol.optcon.filter
::: tfopt.evol.utils.iodata
::: tfopt.evol.utils.params

### Gradient-Based Algorithms

::: tfopt.local.config.constants
::: tfopt.local.config.logconf
::: tfopt.local.exporter.plotout
::: tfopt.local.exporter.sheetutils
::: tfopt.local.objfn.minfn
::: tfopt.local.opt.optrun
::: tfopt.local.optcon.construct
::: tfopt.local.optcon.filter
::: tfopt.local.utils.iodata
::: tfopt.local.utils.params

## Fitting Analysis

::: tfopt.fitanalysis.helper

## ODE Modelling & Parameter Estimation

### Configuration
 
::: config.cli
::: config.config
::: config.constants
::: config.logconf

### Core Functions

::: paramest.normest
::: paramest.toggle

### Weights for Curve Fitting

::: models.weights

### Parameter Estimation

::: paramest.core

### Confidence Intervals using Linearization

::: paramest.identifiability.ci

### Knockout Analysis

::: knockout.helper

### Perturbation & Parameter Sensitivity Analysis

::: sensitivity.analysis

### Model Diagram

::: models.diagram.helpers

### Protein Wise Model Types

::: models.distmod
::: models.randmod
::: models.succmod

### Steady-State Calculation

::: steady.initdist
::: steady.initrand
::: steady.initsucc

### Plotting

::: plotting.plotting

### Utility Functions

::: utils.display
::: utils.tables
::: utils.latexit 

## Global ODE Model

### Core Data Structures & Topology

::: global_model.network
::: global_model.buildmat
::: global_model.params

### Configuration & Data Loading

::: global_model.config
::: global_model.io

### Physics Kernels (JIT)

::: global_model.models

### Numerical Integration & Solvers

::: global_model.simulate
::: global_model.solvers
::: global_model.jacspeedup
::: global_model.steadystate
::: global_model.model_ivp

### Optimization & Loss Functions

::: global_model.optproblem
::: global_model.lossfn
::: global_model.optuna_solver
::: global_model.runner
::: global_model.refine
::: global_model.scan

### Analysis & Visualization

::: global_model.sensitivity
::: global_model.analysis
::: global_model.export
::: global_model.dashboard_app
::: global_model.dashboard_bundle

### Utilities

::: global_model.utils
::: global_model.cache
