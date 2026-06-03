# Global Networkmodel Workflow

## Current scope

The global workflow is the `networkmodel.runner` path. It loads the configured network and data files, creates `Index`, `KinaseInput`, and `System` objects, prepares loss arrays, and solves a scalar objective.

## Optimization path

The scalar objective is represented by `GlobalODEScalarObjective`. Optimization runs through `jaxopt.ProjectedGradient` with `projection=project_bounds`.

## Loss terms

The JAX loss code reports `mrna_loss`, `protein_loss`, and `phospho_loss`. Each available layer contributes weighted mean squared error to the scalar total.

## Parameters

The optimizer slice layout contains `A_i`, `B_i`, `C_i`, `D_i`, `E_i`, `c_k`, `tf_scale`, and `Dp_i`.

## What this module does not do

This page does not describe separate per-protein fitting as part of this networkmodel path. It does not document optimizer choices outside `jaxopt.ProjectedGradient`.
