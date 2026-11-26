# Post-Processing Scripts

The `scripts/` directory contains stand-alone analysis and visualization utilities that operate on
the outputs of the main pipeline. They are not part of the library packages. Run them through pixi
from the repository root (`pixi run python scripts/<folder>/<script>.py`); pixi activation puts the
repository root on `PYTHONPATH`. Outside pixi, set `PYTHONPATH` to the repository root yourself.
See `scripts/README.md` and the README in each `scripts/` subfolder.

---

## When to Run These Scripts

Run scripts **after** the main pipeline has produced result files (Excel exports, CSV predictions,
JSON parameters). The scripts require outputs from `networkmodel` or the result workbooks of the
external KinOpt and TFOpt tools (see [External tools](external_tools.md)).

---

## Script Reference

### `analyze_tf_kin_counts.py`

**Purpose:** Analyze TF and kinase counts in beta values and target gene data from optimization
results. Computes phosphosite (PSite) statistics per entity.

**When to run:** After the external TFOpt and/or KinOpt tools have produced their results Excel files.

**Inputs:**
- `data/tfopt_results.xlsx` (default) — TF optimization results
- `data/kinopt_results.xlsx` (default) — Kinase optimization results

**Outputs:**
- CSV files with PSite count statistics, saved to `results_scripts/` (default)

**Example:**
```bash
pixi run python scripts/analysis/analyze_tf_kin_counts.py \
  --tfopt-xlsx data/tfopt_results.xlsx \
  --kinopt-xlsx data/kinopt_results.xlsx \
  --out-dir results_scripts
```

All three arguments are optional; the values above are the defaults.

---

### `compare_mechanisms.py`

**Purpose:** Interactive Streamlit application for comparing network mechanisms across different
model topologies. Visualizes kinase-substrate and TF-gene networks with global knockout effects.

**When to run:** After `networkmodel` has produced results.

**Inputs:** Uses `networkmodel.config` defaults (reads `config.toml`).

**Outputs:** Interactive Streamlit dashboard (browser).

**Example:**
```bash
pixi run streamlit run scripts/legacy/compare_mechanisms.py
```

> **Note:** Requires `streamlit`, `gravis`, `networkx`, and `imageio` in addition to the standard
> PhosKinTime dependencies.

---

### `curve_similarity.py`

**Purpose:** Compute per-row discrete Fréchet distances between "Observed" and "Estimated" curves
from `tfopt_results.xlsx` and `kinopt_results.xlsx`. Useful for ranking best/worst model fits.

**When to run:** After the external TFOpt or KinOpt tools have produced their results Excel files.

**Inputs:**
- Excel file(s) with sheets named `"Observed"` and `"Estimated"` (wide format: one row per
  entity, columns are time points)

**Outputs:**
- CSV file with per-row Fréchet scores (lower = better fit)

**Example:**
```bash
pixi run python scripts/analysis/curve_similarity.py \
  --tfopt-xlsx data/tfopt_results.xlsx \
  --kinopt-xlsx data/kinopt_results.xlsx \
  --out-dir results_scripts
```

**Interpretation:**
- Fréchet distance = minimum "leash length" to walk both curves in order
- Lower is better; 0 means identical curves
- Scale-dependent: only compare values within the same dataset/scale

---

### `export_subnetworks.py`

**Purpose:** Export per-shared-node k-hop subnetworks from the kinase network (`input2`) and
TF network (`input4`). Useful for focused network analysis around specific proteins.

**When to run:** After preprocessing (input files must exist in `data/`).

**Inputs:**
- `input2.csv` — kinase-substrate network (columns such as `GeneID`, `Psite`, and `Kinase`)
- `input4.csv` — TF-gene edges (columns such as `Source` and `Target`)

**Outputs:**
- `<outdir>/<NODE>/<NODE>_input2.csv` — subnetwork slice for each shared node
- `<outdir>/<NODE>/<NODE>_input4.csv`
- `<outdir>/INDEX.csv` — summary of all nodes with counts and max hop distance
- `<outdir>.zip` (optional) — zipped archive

**Example:**
```bash
pixi run python scripts/analysis/export_subnetworks.py \
  --input2 data/input2.csv \
  --input4 data/input4.csv \
  --outdir results_subnetworks \
  --hops 2
```

Use `--hops auto` to extract the full weakly connected component instead of a fixed hop radius.

---

### `find_protein_accumulators.py`

**Purpose:** Identify "accumulator" proteins where predicted protein fold-change greatly exceeds
predicted mRNA fold-change (ratio > 100), suggesting post-transcriptional regulation or high
protein stability.

**When to run:** After `networkmodel` has produced predicted protein and RNA files.

**Inputs:**
- `pred_prot_picked.csv` — predicted protein fold-change (from `networkmodel` output)
- `pred_rna_picked.csv` — predicted RNA fold-change (from `networkmodel` output)

**Outputs:**
- Printed to stdout: accumulator proteins with their coupling ratios. No files are written.

**Example:**
```bash
pixi run python scripts/analysis/find_protein_accumulators.py \
  --prot results_model_global/pred_prot_picked.csv \
  --rna  results_model_global/pred_rna_picked.csv
```

---

### `mechanistic_insights.py`

**Purpose:** Extract four biological insights from the optimized model:

1. **Refractory Period** — flash vs. stable signaling patterns
2. **Kinetic Lag** — time delay between protein signal and RNA response
3. **Transcriptional Saturation** — digital switching behavior
4. **Feedback Gain** — revolving-door feedback loops

**When to run:** After `networkmodel` has produced optimized parameters.

**Inputs:**
- `data/input2.csv` (kinase network)
- `data/input4.csv` (TF network)
- `data/input1.csv` (MS/protein data)
- `data/input3.csv` (RNA data)
- `data/input1.csv` (phospho data, can overlap with MS)
- KinOpt and TFOpt result workbooks (`--kinopt`, `--tfopt`; Excel `.xlsx`, defaults from `config.toml`)
- `--output-dir` (required): a networkmodel results folder containing `fitted_params_picked.json`

**Outputs:**
- `<output-dir>/Mechanistic_Discovery_Report.xlsx` (one sheet per insight)

**Example:**
```bash
pixi run python scripts/analysis/mechanistic_insights.py \
  --kinase-net data/input2.csv \
  --tf-net     data/input4.csv \
  --ms         data/input1.csv \
  --rna        data/input3.csv \
  --phospho    data/input1.csv \
  --kinopt     data/kinopt_results.xlsx \
  --tfopt      data/tfopt_results.xlsx \
  --output-dir results_model_global
```

---

### `temporal_sensitivity.py`

**Purpose:** Global sensitivity analysis (GSA) using Sobol' indices and Saltelli sampling
(via SALib). Identifies the most influential parameters over time in the global model.

**When to run:** After `networkmodel` optimization has completed; requires result files in
`--results-dir`.

**Inputs:**
- RNA data, kinase/TF network files (from `config.toml` defaults)
- JSON parameter files in `--results-dir`

**Outputs:**
- Plotly HTML plots for parameter rankings over time:
  `<results-dir>/global_sensitivity_analysis/gsa_<protein>.html`

**Example:**
```bash
pixi run python scripts/analysis/temporal_sensitivity.py \
  --results-dir results_model_global \
  --samples 128
```

Increase `--samples` for more accurate Sobol' estimates (must be a power of 2; default 128).
High sample counts significantly increase computation time.

---

## Notes

- Run scripts from the repository root: default input and output paths are relative to the
  current working directory.
- Scripts that use argparse defaults rely on `config.toml`-based path constants from
  `networkmodel/config.py`.
- The dependencies of `compare_mechanisms.py` (`streamlit`, `gravis`, `networkx`, `imageio`) are in
  the default pixi environment (`pixi.toml`).

## Dashboard-ready output directories

Major backend workflows accept an explicit `--outdir`/`--output-dir` option so automated tools and the no-code dashboard can discover outputs consistently. A run directory contains `metadata.json`, `command.txt`, `console.log`, `config_resolved.yaml` when applicable, and the standard `tables/`, `plots/`, `logs/`, `reports/`, and `artifacts/` subdirectories. Legacy filenames are retained at the run-directory root where existing scripts or documentation rely on them, with dashboard-facing copies mirrored into the standard subfolders.

## Unified dashboard integration and legacy script status

The unified dashboard provides workflow-specific result panels for KinOpt and TFOpt result workbooks (produced by the external tools; see [External tools](external_tools.md)), ProtWise, Networkmodel, and advanced analyses. Analysis scripts remain callable backends; the dashboard builds parameterized commands and writes outputs under the selected run directory instead of duplicating scientific computations in Streamlit.

## `export_model_to_sbml.py` (`scripts/export/`)

Exports a fitted **distributive** networkmodel run as reaction-based SBML. It reconstructs the
fitted model directly from the run's exported topology and input matrices, not from the current
`config.toml` data paths or the posterior sampling payload. See also `scripts/export/README.md`.

### Required saved files

All must exist in the results folder passed as `--results` (for example
`results_network_distributive/`):

- `fitted_params_picked.json`
- `model_parameters_genes.csv`
- `model_parameters_genes_psites.csv`
- `model_parameters_kinases.csv`
- `network_W_global.csv`
- `network_tf_mat.csv`
- `network_kinase_inputs.csv`
- `S_rates_picked.csv`
- `initial_conditions_y0.csv`

The script refuses to invent missing sites, initial states, kinase drivers or weights.

### Usage

Run from the repository root:

```bash
pixi run python scripts/export/export_model_to_sbml.py \
  --results results_network_distributive \
  --conf config.toml \
  --check-only

pixi run python scripts/export/export_model_to_sbml.py \
  --results results_network_distributive \
  --conf config.toml \
  --out results_network_distributive/sbml_reactions
```

The `--conf` file must select the original **distributive (MODEL=0) RHS**. It is used for the repository's import-time model selection, but its *input data paths* are not used to reconstruct the fitted topology.

The output XML is `results_network_distributive/sbml_reactions/networkmodel_reactions.xml`. To overwrite an existing XML use `--overwrite` explicitly. `--check-only` does not write or modify SBML.

### Validation

```bash
pixi run -e dev pytest -q tests/test_export_model_to_sbml.py
```

The exporter cross-checks:

1. Saved parameters against the ordered gene, site, and kinase tables.
2. Saved matrix edges and weights, and gene names.
3. Sample-and-hold kinase inputs against the fitted phosphorylation-drive trajectories.
4. Initial state mapping against the long-form saved initial-condition table.
5. The exported reaction graph's evaluated ODE against `networkmodel.backend.make_networkmodel_rhs`, the JAX fitting RHS. The legacy NumPy `System.rhs` has a different TF-input squashing path, so it is not the reference for numerical equivalence.
6. libSBML consistency and SBML file round-trip, when `python-libsbml` is installed.

The numerical check evaluates selected time points and one representative nonnegative state vector; it is a regression preflight, **not** a formal mathematical proof or a complete simulation validation.

For a fitted result with 17 modeled proteins/genes, 36 phosphosites, and 27 kinases, the reaction-based SBML should contain 176 reactions and 97 species. The actual script prints the recovered counts; never force these counts if a check fails.

### Scope and limitations

- Only the **distributive** networkmodel kinetics are supported. Other topologies (`sequential`, `combinatorial`, `saturating`) require separate reaction definitions and tests.
- Exported coefficients represent the **picked fitted solution**, not the posterior mean.
- Only numeric (post-proxy) TF network coefficients survive in the saved exported TF matrix. Original TF names that were proxied are not recoverable from those matrix exports alone. The **effective model** remains reconstructable.
- The SBML contains a time-dependent kinase boundary input as a `piecewise` assignment, explicit catalytic/reaction modifiers, and reaction kinetics for RNA, protein, and phosphosite species.
