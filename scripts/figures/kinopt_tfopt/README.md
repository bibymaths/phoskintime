# scripts/figures/kinopt_tfopt/

These are KinOpt/TFOpt network readouts, publication-style figures and schematic diagrams.
- **No CLI.** None of these scripts accepts command-line arguments. Input and output paths are constants at the top of each file, and they are relative to the current working directory, so run them from the repository root (see [`../../README.md`](../../README.md)).
- **Interactive versions.** The same readouts are re-implemented interactively in `scripts/apps/kinopt_results_app.py` and `scripts/apps/tfopt_results_app.py`.

| Script | What it does | Inputs (constants) | Outputs (constants) |
|---|---|---|---|
| `kinopt_network_readout.py` | Readout of the KinOpt network using the fitted α/β model equation: kinase load, dominant kinases per target, knockout effects. | `data/kinopt_results.xlsx` | `results_scripts/figures_kinopt/`: `kinopt_kinase_load.csv`, `kinopt_target_dominant_kinases.csv`, `kinopt_knockout_effects.csv`, `kinopt_latent_kinase_activity_summary.csv` |
| `kinopt_network_viz.py` | Figures from the KinOpt readout CSVs. | the CSVs above (run `kinopt_network_readout.py` first) | PNG figures in `results_scripts/figures_kinopt/` |
| `tfopt_network_readout.py` | Readout of the TFOpt network: TF load, dominant TFs per target, knockout effects, TF phosphosite statistics. | `data/tfopt_results.xlsx`, `data/input1.csv`, `data/input4.csv` | `results_scripts/figures_tfopt/`: `tfopt_tf_load.csv`, `tfopt_target_dominant_tfs.csv`, `tfopt_knockout_effects.csv`, `tfopt_tf_psite_stats.csv` |
| `tfopt_network_viz.py` | Figures from the TFOpt readout CSVs. | the CSVs above (run `tfopt_network_readout.py` first), `data/input4.csv` | PNG figures in `results_scripts/figures_tfopt/` (also PDF if `SAVE_PDF = True` is set in the file) |
| `make_kinopt_diagram.py` | Graphviz schematics of the KinOpt/TFOpt/global optimisation, with and without constraints. Uses `pygraphviz` if installed (the `viz` pixi environment), otherwise `pydot`. | none | Seven PNGs in the **current working directory**: `kinopt_diagram.png`, `kinopt_constraints.png`, `tfopt_diagram.png`, `tfopt_constraints.png`, `global_diagram.png`, `global_constraints.png`, `kin_tf_opt_diagram.png` (git-ignored at the repository root) |

Example:

```bash
pixi run python scripts/figures/kinopt_tfopt/kinopt_network_readout.py
pixi run python scripts/figures/kinopt_tfopt/kinopt_network_viz.py
```

Which publication figures these scripts produced: TODO(owner).
