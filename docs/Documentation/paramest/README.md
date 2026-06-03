# Parameter Estimation

## Current scope

The current networkmodel parameter-estimation path uses `GlobalODEScalarObjective` and `optimize_scalar_objective`. The optimizer is `jaxopt.ProjectedGradient` with `projection=project_bounds`.

## Stopping criteria

`optimize_scalar_objective` passes `tol` and `maxiter` to JAXopt. The optimizer stops when the distance between iterates is below `tol` or when `maxiter` is reached.

## What this module does not do

This page does not document parameter-estimation algorithms outside the current JAXopt path.
