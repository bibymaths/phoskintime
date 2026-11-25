# scripts/analysis/

These are the analysis backends behind the dashboard's "Advanced analysis" panel (`dashboard/workflow_panels/analysis.py`). The panel only *builds and shows* the command; it does not run it. Run the commands from the repository root (see [`../README.md`](../README.md)).

| Script | What it does | Inputs | Outputs (defaults) |
|---|---|---|---|
| `analyze_tf_kin_counts.py` | Counts phosphosites (β values) per TF and per kinase, and TFs/kinases per target gene, from the TFOpt and KinOpt result workbooks. | `--tfopt-xlsx` (default `data/tfopt_results.xlsx`), `--kinopt-xlsx` (default `data/kinopt_results.xlsx`) | `--out-dir` (default `results_scripts/`): `tf_beta_psite_counts.csv`, `kin_beta_psite_counts.csv`, `per_gene_num_tfs_num_kinases.csv` |
| `curve_similarity.py` | Computes the discrete Fréchet distance, per row, between the "Observed" and "Estimated" sheets of both workbooks (uses `common.frechet`). | `--tfopt-xlsx`, `--kinopt-xlsx` (same defaults as above) | `--out-dir` (default `results_scripts/`): `frechet_tfopt.csv`, `frechet_kinopt.csv` |
| `export_subnetworks.py` | For each node that appears in both networks, exports a k-hop subnetwork (or the whole weakly connected component with `--hops auto`) in the original `input2`/`input4` layouts. | `--input2` (kinase table), `--input4` (TF edge list), all required | `--outdir/<NODE>/<NODE>_input2.csv`, `<NODE>_input4.csv`, `<NODE>_graph.png`, `INDEX.csv`; with `--zip`, also `<outdir>.zip` |
| `find_protein_accumulators.py` | Flags proteins whose ratio of maximum predicted protein fold change to maximum predicted mRNA fold change is above `--threshold` (default 100). | `--prot`, `--rna`: networkmodel prediction CSVs (e.g. `pred_prot_picked.csv`, `pred_rna_picked.csv`), required | Prints to stdout only; writes no files |
| `mechanistic_insights.py` | Computes four descriptive readouts from a fitted networkmodel run (refractory period, kinetic lag, transcriptional saturation, feedback gain). | `--output-dir` (required): a networkmodel results folder containing `fitted_params_picked.json`. Network and data paths default to the values in `config.toml` (`--kinase-net`, `--tf-net`, `--ms`, `--rna`, `--phospho`, `--kinopt`, `--tfopt`). | `<output-dir>/Mechanistic_Discovery_Report.xlsx` |
| `temporal_sensitivity.py` | Sobol global sensitivity analysis over time (SALib, Saltelli sampling) on a fitted networkmodel run. | `--results-dir` (required): a networkmodel results folder containing `fitted_params_picked.json`; `--samples` (default 128) | `<results-dir>/global_sensitivity_analysis/gsa_<protein>.html` |

Example:

```bash
PYTHONPATH=. pixi run python scripts/analysis/curve_similarity.py --out-dir results_scripts
```
