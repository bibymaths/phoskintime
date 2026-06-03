# Metrics

## Current loss terms

`multimodal_loss_from_trajectory` computes layer losses for available observed data. The reported terms are `mrna_loss`, `protein_loss`, and `phospho_loss`.

## Loss calculation

Each active layer uses weighted mean squared error between simulated and observed values. The scalar objective adds the active layer terms and an optional prior penalty when defaults and a prior weight are supplied.

## What this module does not do

This page does not document metrics that are absent from the current networkmodel loss code. It does not describe external ranking methods as part of the scalar objective.
