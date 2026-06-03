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
::: networkmodel.buildmat
::: networkmodel.params

### Configuration & Data Loading

::: networkmodel.config
::: networkmodel.io

### Physics Kernels (JIT)

::: networkmodel.models

### Numerical Integration & Solvers

::: networkmodel.simulate
::: networkmodel.jax_backend
::: networkmodel.jacspeedup
::: networkmodel.steadystate

### Optimization & Loss Functions

::: networkmodel.optproblem
::: networkmodel.lossfn
::: networkmodel.inference
::: networkmodel.runner
::: networkmodel.scan

### Analysis & Visualization

::: networkmodel.sensitivity
::: networkmodel.analysis
::: networkmodel.export
::: networkmodel.dashboard_app
::: networkmodel.dashboard_bundle
::: networkmodel.mode_outputs

### Utilities

::: networkmodel.utils
::: networkmodel.cache
