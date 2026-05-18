# Dashboard Usage

PhosKinTime includes an interactive Streamlit-based dashboard for visualizing the results of the
global network model optimization.

---

## Components

### `global_model/dashboard_app.py`

The main Streamlit application. It loads result files from a specified output directory and renders:

- **Pareto frontier** (3D scatter: protein MSE × RNA MSE × phospho MSE)
- **Convergence history** (fitness over generations)
- **Goodness of fit** (predicted vs. observed for protein, RNA, phospho)
- **Residuals analysis**
- **Image/video galleries** of exported plots (PNG, JPEG, MP4)
- **PDF report viewer**

### `global_model/dashboard_bundle.py`

Saves and loads a compact binary bundle (`dashboard_bundle.pkl`) containing:

| Field | Description |
|---|---|
| `args` | Command-line arguments from the runner |
| `picked_index` | Index of the selected Pareto solution |
| `frechet_scores` | Fréchet distance scores per protein |
| `lambdas` | Lambda regularization values |
| `solver_times` | ODE solver timing per protein |
| `defaults` | Default parameter values |
| `slices` | Parameter slices |
| `xl` / `xu` | Lower / upper parameter bounds |

The bundle avoids pickling complex Pymoo objects, making it portable across Python sessions.

### `run_dashboard.py`

Root-level launcher script. Sets up `sys.path` and calls `global_model.dashboard_app.main()`.

---

## Required Result Files

The dashboard looks for files in the `--output-dir` you specify. Required / optional:

| File | Status | Description |
|---|---|---|
| `dashboard_bundle.pkl` | Required | Saved by `global_model/runner.py` after optimization |
| `pareto_F.csv` | Optional (falls back to bundle) | Pareto objective values |
| `convergence_history.csv` | Optional | Generation-by-generation convergence |
| `pred_prot_picked.csv` | Optional | Predicted protein time series (picked solution) |
| `pred_rna_picked.csv` | Optional | Predicted RNA time series (picked solution) |
| `pred_phospho_picked.csv` | Optional | Predicted phospho time series (picked solution) |
| `*.png` / `*.jpg` | Optional | Any image outputs from `global_model/export.py` |
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

### Option 2: Via the phoskintime-global entry point

`phoskintime-global` runs `global_model/runner.py`. All settings default to `config.toml` values
and can be overridden via CLI arguments:

```bash
# Run with defaults from config.toml
phoskintime-global

# Override specific settings
phoskintime-global \
  --output-dir results_global \
  --n-gen 500 \
  --pop 200 \
  --solver pymoo
```

Key arguments (all optional — defaults come from `config.toml`):

| Argument | Description |
|---|---|
| `--kinase-net` | Path to kinase-substrate network |
| `--tf-net` | Path to TF-gene network |
| `--ms` | Path to MS protein data |
| `--rna` | Path to RNA data |
| `--output-dir` | Output directory |
| `--n-gen` | Number of generations |
| `--pop` | Population size |
| `--solver` | `pymoo` or `optuna` |
| `--sensitivity` | Enable sensitivity analysis |
| `--refine` | Enable refinement pass |
| `--scan` | Run hyperparameter scan |

This runs the full optimization pipeline via `global_model/runner.py` and saves the bundle.
After it finishes, launch the dashboard with Option 1 above.

### Option 3: Python script

```bash
python run_dashboard.py --output-dir results_model_global_distributive_knockout
```

---

## Common Failure Modes

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: dashboard_bundle.pkl` | Run `phoskintime-global` first to generate results |
| `KeyError` on bundle fields | Bundle was saved by an older version; re-run the optimizer |
| Streamlit not found | Install streamlit: `pip install streamlit` |
| Empty plots | Result CSVs are missing; check `--output-dir` path |

---

## Default Port

Streamlit defaults to port **8501**. To use a different port:

```bash
streamlit run run_dashboard.py --server.port 8502 -- --output-dir ...
```

---

## Relationship Between `dashboard_app.py` and `dashboard_bundle.py`

- `dashboard_bundle.py` is a **save/load utility** — it knows nothing about Streamlit.
  It is called by `global_model/runner.py` at the end of optimization to persist results.
- `dashboard_app.py` is the **Streamlit UI** — it calls `dashboard_bundle.py`'s `load_dashboard_bundle()`
  at startup to restore the persisted results.

This separation ensures the dashboard can be launched independently from the optimization run,
and makes the bundle serialization independent of UI concerns.
