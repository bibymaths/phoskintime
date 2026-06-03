# Models

## Current public model kernels

The model kernels are implemented in `models.py`. The public kernels are `saturating_rhs`, `distributive_rhs`, `sequential_rhs`, and `combinatorial_rhs`.

## State construction

`Index` maps proteins and phosphorylation sites to state-vector blocks. `System` selects the model kernel through the configured internal model code.

## Parameters

The public parameter layout contains `A_i`, `B_i`, `C_i`, `D_i`, `E_i`, `c_k`, `tf_scale`, and `Dp_i`.

## What this module does not do

This page does not describe model kernels that are absent from `models.py`. It does not document unsupported solver backends.
