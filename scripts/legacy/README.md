# scripts/legacy/

These are older tools, kept for reference. The unified dashboard (`pixi run dashboard`) covers most interactive use.

## `compare_mechanisms.py`

A Streamlit app for knockout simulations and mechanism comparison on a fitted networkmodel run.

- **No CLI.** The run it analyses is set by two constants at the top of the file:
  - `RESULTS_DIR_PICKED = Path("./results_network_combinatorial")` (relative to the current working directory);
  - `INTENDED_MODEL = 2` (combinatorial; `0` distributive, `1` sequential, `4` saturating).

  To analyse a different run, edit these constants.
- **Inputs:** files from that results folder, e.g. `fitted_params_picked.json`, `S_rates_picked.csv`, `pred_prot_picked.csv`, `pred_rna_picked.csv`, `pred_phospho_picked.csv`.
- **Outputs:** CSV downloads offered in the app (knockout deltas, edge-dynamics and sweep tables).
- **Start** from the repository root (see [`../README.md`](../README.md)):

```bash
pixi run streamlit run scripts/legacy/compare_mechanisms.py
```
