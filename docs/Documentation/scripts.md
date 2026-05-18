# Post-Processing Scripts

The `scripts/` directory contains stand-alone analysis and visualization utilities that operate on
the outputs of the main pipeline. These are not part of the installable package and are intended to
be run directly with Python from the project root.

---

## When to Run These Scripts

Run scripts **after** the main pipeline has produced result files (Excel exports, CSV predictions,
JSON parameters). The scripts require outputs from `kinopt`, `tfopt`, or `global_model`.

---

## Script Reference

### `analyze_tf_kin_counts.py`

**Purpose:** Analyze TF and kinase counts in beta values and target gene data from optimization
results. Computes phosphosite (PSite) statistics per entity.

**When to run:** After `tfopt` and/or `kinopt` have produced their results Excel files.

**Inputs:**
- `data/tfopt_results.xlsx` (default) — TF optimization results
- `data/kinopt_results.xlsx` (default) — Kinase optimization results

**Outputs:**
- CSV files with PSite count statistics, saved to `results_scripts/` (default)

**Example:**
```bash
python scripts/analyze_tf_kin_counts.py
```

> **Note:** `analyze_tf_kin_counts.py` does not accept command-line arguments. It uses hardcoded
> default paths:
> - Input: `data/tfopt_results.xlsx`, `data/kinopt_results.xlsx`
> - Output: `results_scripts/`
>
> To change input paths, edit the `main()` call at the bottom of the script or import and call
> `main(tfopt_xlsx=..., kinopt_xlsx=..., out_dir=...)` from your own script.

---

### `compare_mechanisms.py`

**Purpose:** Interactive Streamlit application for comparing network mechanisms across different
model topologies. Visualizes kinase-substrate and TF-gene networks with global knockout effects.

**When to run:** After `global_model` has produced results.

**Inputs:** Uses `global_model.config` defaults (reads `config.toml`).

**Outputs:** Interactive Streamlit dashboard (browser).

**Example:**
```bash
streamlit run scripts/compare_mechanisms.py
```

> **Note:** Requires `streamlit`, `gravis`, `networkx`, and `imageio` in addition to the standard
> PhosKinTime dependencies.

---

### `curve_similarity.py`

**Purpose:** Compute per-row discrete Fréchet distances between "Observed" and "Estimated" curves
from `tfopt_results.xlsx` and `kinopt_results.xlsx`. Useful for ranking best/worst model fits.

**When to run:** After `tfopt` or `kinopt` optimization.

**Inputs:**
- Excel file(s) with sheets named `"Observed"` and `"Estimated"` (wide format: one row per
  entity, columns are time points)

**Outputs:**
- CSV file with per-row Fréchet scores (lower = better fit)

**Example:**
```bash
python scripts/curve_similarity.py \
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
- `input2.csv` — kinase-substrate network (columns: `GeneID`, `Psite`, `Kinase`, ...)
- `input4.csv` — TF-gene edges (columns: `Source`, `Target`, ...)

**Outputs:**
- `<outdir>/<NODE>/<NODE>_input2.csv` — subnetwork slice for each shared node
- `<outdir>/<NODE>/<NODE>_input4.csv`
- `<outdir>/INDEX.csv` — summary of all nodes with counts and max hop distance
- `<outdir>.zip` (optional) — zipped archive

**Example:**
```bash
python scripts/export_subnetworks.py \
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

**When to run:** After `global_model` has produced predicted protein and RNA files.

**Inputs:**
- `pred_prot_picked.csv` — predicted protein fold-change (from `global_model` output)
- `pred_rna_picked.csv` — predicted RNA fold-change (from `global_model` output)

**Outputs:**
- CSV file listing accumulator proteins with their coupling ratios

**Example:**
```bash
python scripts/find_protein_accumulators.py \
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

**When to run:** After `global_model` has produced optimized parameters.

**Inputs:**
- `data/input2.csv` (kinase network)
- `data/input4.csv` (TF network)
- `data/input1.csv` (MS/protein data)
- `data/input3.csv` (RNA data)
- `data/input1.csv` (phospho data, can overlap with MS)
- JSON fitted parameter files from `kinopt` and `tfopt`

**Outputs:**
- CSV and plot files in `--output-dir` (default: `output/`)

**Example:**
```bash
python scripts/mechanistic_insights.py \
  --kinase-net data/input2.csv \
  --tf-net     data/input4.csv \
  --ms         data/input1.csv \
  --rna        data/input3.csv \
  --phospho    data/input1.csv \
  --kinopt     data/fitted_params_picked.json \
  --tfopt      data/fitted_params_picked.json \
  --output-dir results_insights
```

---

### `temporal_sensitivity.py`

**Purpose:** Global sensitivity analysis (GSA) using Sobol' indices and Saltelli sampling
(via SALib). Identifies the most influential parameters over time in the global model.

**When to run:** After `global_model` optimization has completed; requires result files in
`--results-dir`.

**Inputs:**
- RNA data, kinase/TF network files (from `config.toml` defaults)
- JSON parameter files in `--results-dir`

**Outputs:**
- CSV file with Sobol' first-order (`S1`) and total-order (`ST`) sensitivity indices
- Plotly HTML plots for parameter rankings over time

**Example:**
```bash
python scripts/temporal_sensitivity.py \
  --results-dir results_model_global \
  --samples 128
```

Increase `--samples` for more accurate Sobol' estimates (must be a power of 2; default 128).
High sample counts significantly increase computation time.

---

## Notes

- Scripts that import from `global_model` must be run from the project root directory.
- Scripts that use argparse defaults rely on `config.toml`-based path constants from
  `global_model/config.py`.
- The `compare_mechanisms.py` script requires additional dependencies (`gravis`, `networkx`,
  `imageio`) not included in the base `requirements.txt`.
