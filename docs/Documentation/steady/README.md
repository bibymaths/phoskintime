# Steady-State and Initial Conditions

## Current scope

The steady-state and initial-condition source is split across `analysis.py` and `steadystate.py`. `simulate_until_steady` simulates a system over a long time grid, and `build_y0_from_data` creates initial state values from observed baseline data.

## Current public API

`plot_steady_state_all` writes diagnostic plots for a simulated trajectory. `simulate_diffrax` provides the Diffrax-backed simulation helper used by the current path.

## What this module does not do

This page does not document alternate ODE integration backends. It does not describe initial-condition features absent from `steadystate.py`.
