# Architecture and Data Flow

## Current networkmodel workflow

The current global workflow starts in `runner.py`. The runner loads data, builds topology, prepares loss arrays, runs the scalar objective, and writes outputs.

```text
runner -> optproblem -> jax_backend
runner -> inference -> jax_backend
runner -> sensitivity
runner -> export / analysis / dashboard
```

## Optimization and simulation

The optimizer path uses `jaxopt.ProjectedGradient` with `projection=project_bounds`. The ODE path uses `diffrax.Kvaerno4` or `diffrax.Kvaerno5` through `DiffraxSolverConfig`.

## What this module does not do

This page does not describe alternate optimizer backends. It does not document modules outside the current `networkmodel` source inventory. It does not describe a data-flow edge unless that edge appears in the import graph above.
