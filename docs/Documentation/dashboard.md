# Dashboard Usage

PhosKinTime includes an interactive Streamlit-based dashboard for visualizing the results of the
global network model optimization.

---

## Components

### `networkmodel/dashboard_app.py`

The main Streamlit application. It loads result files from a specified output directory and renders:

- **Scalar objective diagnostics** for the JAXopt/Diffrax run
- **Convergence history** (fitness over generations)
- **Goodness of fit** (predicted vs. observed for protein, RNA, phospho)
- **Residuals analysis**
- **Image/video galleries** of exported plots (PNG, JPEG, MP4)
- **PDF report viewer**

### `networkmodel/dashboard_bundle.py`

Saves and loads a compact binary bundle (`dashboard_bundle.pkl`) containing:

| Field | Description |
|---|---|
| `args` | Command-line arguments from the runner |
| `picked_index` | Index of the selected scalar-objective solution |
| `frechet_scores` | Fréchet distance scores per protein |
| `lambdas` | Lambda regularization values |
| `solver_times` | ODE solver timing per protein |
| `defaults` | Default parameter values |
| `slices` | Parameter slices |
| `xl` / `xu` | Lower / upper parameter bounds |

The bundle stores scalar objective arrays and mode metadata, making it portable across Python sessions.

### `run_dashboard.py`

Root-level launcher script. Sets up `sys.path` and calls `networkmodel.dashboard_app.main()`.

---

## Required Result Files

The dashboard looks for files in the `--output-dir` you specify. Required / optional:

| File | Status | Description |
|---|---|---|
| `dashboard_bundle.pkl` | Required | Saved by `networkmodel/runner.py` after optimization |
| `scalar_objective.csv` | Preferred | Scalar objective values with mode metadata |
| `pareto_F.csv` | Backward-compatible alias | Scalar objective values |
| `convergence_history.csv` | Optional | Generation-by-generation convergence |
| `pred_prot_picked.csv` | Optional | Predicted protein time series (picked solution) |
| `pred_rna_picked.csv` | Optional | Predicted RNA time series (picked solution) |
| `pred_phospho_picked.csv` | Optional | Predicted phospho time series (picked solution) |
| `*.png` / `*.jpg` | Optional | Any image outputs from `networkmodel/export.py` |
| `*.mp4` | Optional | Convergence animation |
| `*.pdf` | Optional | Report PDFs |

If `dashboard_bundle.pkl` is missing the dashboard will fail with a `FileNotFoundError`.

---

## How to Launch

### Option 1: Direct Streamlit invocation (recommended)

```bash
streamlit run run_dashboard.py -- --output-dir results_model_global_distributive_knockout
```

Streamlit will start a local web server. The default URL is:

```
http://localhost:8501
```

### Option 2: Via the networkmodel runner entry point

`python -m networkmodel.runner` runs `networkmodel/runner.py`. All settings default to `config.toml` values
and can be overridden via CLI arguments:

```bash
# Run with defaults from config.toml
python -m networkmodel.runner

# Override specific settings
python -m networkmodel.runner \
  --output-dir results_global \
  --n-gen 500 \
  --solver jaxopt
```

Key arguments (all optional — defaults come from `config.toml`):

| Argument | Description |
|---|---|
| `--kinase-net` | Path to kinase-substrate network |
| `--tf-net` | Path to TF-gene network |
| `--ms` | Path to MS protein data |
| `--rna` | Path to RNA data |
| `--output-dir` | Output directory |
| `--n-gen` | Maximum JAXopt iterations |
| `--solver` | `jaxopt`; legacy values are accepted and mapped with warnings |
| `--sensitivity` | Enable sensitivity analysis |
| `--scan` | Run hyperparameter scan |

This runs the full optimization pipeline via `networkmodel/runner.py` and saves the bundle.
After it finishes, launch the dashboard with Option 1 above.

### Option 3: Python script

```bash
python run_dashboard.py --output-dir results_model_global_distributive_knockout
```

---

## Common Failure Modes

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: dashboard_bundle.pkl` | Run `python -m networkmodel.runner` first to generate results |
| `KeyError` on bundle fields | Bundle was saved by an older version; re-run the optimizer |
| Streamlit not found | Install streamlit: `pip install streamlit` |
| Empty plots | Result CSVs are missing; check `--output-dir` path |

---

## Default Port

Streamlit defaults to port **8501**. To use a different port:

```bash
streamlit run run_dashboard.py --server.port 8502 -- --output-dir results_model_global_distributive_jax
```

---

## Relationship Between `dashboard_app.py` and `dashboard_bundle.py`

- `dashboard_bundle.py` is a **save/load utility** — it knows nothing about Streamlit.
  It is called by `networkmodel/runner.py` at the end of optimization to persist results.
- `dashboard_app.py` is the **Streamlit UI** — it calls `dashboard_bundle.py`'s `load_dashboard_bundle()`
  at startup to restore the persisted results.

This separation ensures the dashboard can be launched independently from the optimization run,
and makes the bundle serialization independent of UI concerns.

## Inference Outputs

The dashboard includes an **Inference** tab when optional inference files are present. It displays:

- `optimization/best_fit.csv`
- `optimization/multistart_summary.csv`
- `optimization/multistart_parameters.csv`
- `profiles/profile_likelihood_summary.csv`
- `posterior/posterior_summary.csv`
- `posterior/posterior_samples.csv`
- plots from `plots/multistart/`, `plots/profile_likelihood/`, and `plots/posterior/`

These are diagnostic scalar-objective summaries, not true multi-objective optimizer fronts.

---

## Unified no-code dashboard

The newer unified dashboard lives in `dashboard/app.py` and can browse existing result directories, launch registered CLI/Pixi workflows, preview uploads, and show workflow-specific panels. See [No-code PhosKinTime dashboard](no_code_dashboard.md) for launch commands, supported workflows, result-directory contract details, upload behavior, examples, and troubleshooting.
