# scripts/apps/

These are stand-alone Streamlit apps, separate from the unified dashboard (`dashboard/app.py`, started with `pixi run dashboard`). Start them from the repository root (see [`../README.md`](../README.md)).

| App | What it shows | Inputs | How to start |
|---|---|---|---|
| `run_dashboard.py` | Launcher for the legacy networkmodel results dashboard (`networkmodel/dashboard_app.py`). | `--output-dir`: a networkmodel results folder (required) | `pixi run streamlit run scripts/apps/run_dashboard.py -- --output-dir <results folder>` |

The same launcher is available as a pixi task: `pixi run network-dashboard --output-dir <results folder>`.

