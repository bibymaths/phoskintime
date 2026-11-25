# scripts/apps/

These are stand-alone Streamlit apps, separate from the unified dashboard (`dashboard/app.py`, started with `pixi run dashboard`). Start them from the repository root (see [`../README.md`](../README.md)).

| App | What it shows | Inputs | How to start |
|---|---|---|---|
| `kinopt_results_app.py` | KinOpt network readout and visualisation, recomputed from a KinOpt result workbook. | A KinOpt XLSX (sheets Observed / Estimated / Alpha Values / Beta Values), uploaded in the app | `PYTHONPATH=. pixi run streamlit run scripts/apps/kinopt_results_app.py` |
| `tfopt_results_app.py` | TFOpt network readout, visualisation and network rendering (gravis), recomputed from a TFOpt result workbook. | `tfopt_results.xlsx`, `input1.csv` and optionally `input3.csv`, all uploaded in the app | `PYTHONPATH=. pixi run streamlit run scripts/apps/tfopt_results_app.py` |
| `run_dashboard.py` | Launcher for the legacy networkmodel results dashboard (`networkmodel/dashboard_app.py`). | `--output-dir`: a networkmodel results folder (required) | `PYTHONPATH=. pixi run streamlit run scripts/apps/run_dashboard.py -- --output-dir <results folder>` |

Note: the `network-dashboard` pixi task currently runs `python scripts/apps/run_dashboard.py` without `--output-dir`. That command does not start a Streamlit server, so use the `streamlit run …` command above.

`kinopt_results_app.py` and `tfopt_results_app.py` were moved here from `app/kinopt.py` and `app/tfopt.py`. They were renamed so that no file in this folder has the same name as the `kinopt`/`tfopt` packages.
