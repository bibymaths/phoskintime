# No-code PhosKinTime dashboard

The unified dashboard is a Streamlit application for browsing completed PhosKinTime runs and for launching existing CLI/Pixi workflows without editing Python code. It is intentionally a thin user interface: scientific fitting, ODE solving, optimization, and analysis remain in the existing backend modules and scripts.

## Launching the dashboard

From the repository root, start the dashboard with Pixi:

```bash
pixi run dashboard
```

For development with Streamlit reload-on-save enabled:

```bash
pixi run dashboard-dev
```

The tasks are defined in `pixi.toml` and run `streamlit run dashboard/app.py` with `PYTHONPATH=.` so the in-repository packages are importable.

## Supported workflows

The launcher registry exposes only structured, known commands. It builds argument lists, not shell strings, and runs subprocesses from the repository root.

| Workflow | What it launches | Typical inputs |
| --- | --- | --- |
| `prep` | preprocessing cleanup task | files referenced by the current project config |
| `kinopt-local` | `python -m kinopt.local` | phosphosite/protein tables, kinase network data, optional config |
| `tfopt-local` | `python -m tfopt.local` | RNA/mRNA tables, TF network data, optional config |
| `protwise-model` | `python -m protwise.runner.main` | protein/phosphosite data and optional config |
| `networkmodel` | `python -m networkmodel.runner` | kinase network, TF network, omics tables, optional config |
| `phoskintime-all` | `python -m config.cli all` | project-level configuration and data files |

The dashboard does not invent new CLI flags. If an option is not listed in the workflow registry, it is not generated.

## Result directory contract

Each dashboard-launched run uses a predictable result directory under `results/<workflow>/<run_id>/` unless a workflow-specific backend chooses a compatible location. The browser expects this contract:

```text
results/<run_id>/
├── metadata.json
├── command.txt
├── console.log
├── config_resolved.yaml
├── tables/
├── plots/
├── logs/
├── reports/
└── artifacts/
```

The result browser also recognizes practical legacy layouts, including Networkmodel top-level CSVs such as `scalar_objective.csv`, `convergence_history.csv`, prediction CSVs, `optimization/`, `profiles/`, `posterior/`, and legacy `kinopt_results.xlsx` or `tfopt_results.xlsx` files.

## Upload behavior

The upload panel accepts `.csv`, `.tsv`, `.xlsx`, `.yaml`, `.yml`, `.json`, and `.txt` files. Uploaded files are copied to a per-run folder:

```text
dashboard_uploads/<run_id>/
```

Uploads are not written into `data/` and are not modified silently. The dashboard checks for empty files, unsupported extensions, unreadable files, and duplicate filenames after sanitization. CSV, TSV, and Excel inputs can be previewed with pandas before they are assigned to a workflow.

Workflow-specific validation follows the backend readers. ProtWise protein input and Networkmodel kinase network, TF network, MS/protein, RNA, and phosphoproteomics inputs must be CSV files because those runners read them with `pandas.read_csv`. Excel remains valid only for workflow inputs that are actually read as Excel, such as ProtWise phosphosite/RNA files or previous KinOpt/TFOpt result workbooks.

## Command preview and execution

Before a run starts, the dashboard shows the exact command preview and the underlying argument list. This keeps runs reproducible and helps users copy the command into a terminal if preferred.

During execution, stdout and stderr are streamed into the page and written to `console.log` in the run directory. The runner records `command.txt` and launcher metadata in `metadata.json`. Failures are displayed with the return code and a log tail rather than crashing the Streamlit session.

## Result browser usage

Use the **Browse results** tab to select an existing result directory. The browser displays:

- `metadata.json` with `st.json` when valid;
- `command.txt`, `console.log`, and other logs as text/code;
- CSV, TSV, and Excel tables with `st.dataframe`;
- PNG, JPG, JPEG, SVG, and HTML plots;
- Markdown, HTML, and PDF reports where Streamlit can render them;
- artifacts as downloadable files;
- a ZIP download button for the selected result directory.

If standard contract files are missing, the browser reports them clearly and still attempts to show recognized legacy outputs.

## Workflow-specific panels

The result browser includes workflow tabs for KinOpt, TFOpt, ProtWise, Networkmodel, and advanced analyses.

- **KinOpt** displays discovered KinOpt result workbooks, alpha/beta-related tables where available, observed-vs-estimated output files, and existing plots.
- **TFOpt** displays TFOpt result workbooks, TF alpha/beta or equivalent tables, latent activity/dominance/knockout outputs when present, and existing plots.
- **ProtWise** displays ODE prediction tables, residual/fits files, sensitivity outputs, and generated plots or reports.
- **Networkmodel** supports both the standardized result layout and legacy Networkmodel outputs such as `dashboard_bundle.pkl`, scalar objective CSVs, convergence history, predictions, optimization summaries, profiles, and posterior outputs.
- **Advanced analysis** exposes parameterized command wrappers for existing scripts such as curve similarity, protein accumulator detection, subnetwork export, mechanistic insights, temporal sensitivity, and mechanism comparison. These analyses are not run automatically on page load.

Optional visualization packages are imported only when the affected panel is used. If a dependency is unavailable, the dashboard should show a clear message instead of failing during import.

## Examples

### Browse an existing result directory

1. Run `pixi run dashboard`.
2. Open **Browse results**.
3. Enter the base results folder, for example `results`.
4. Select a run folder.
5. Review metadata, tables, plots, logs, reports, artifacts, and workflow-specific tabs.

### Run KinOpt

1. Open **Run workflow**.
2. Select `KinOpt local (kinopt-local)`.
3. Upload or select phosphosite/protein and kinase-network inputs as required by the current CLI/config.
4. Preview uploaded tables.
5. Confirm validation passes.
6. Review the command preview.
7. Click **Run workflow**.
8. When the process completes, the result browser opens the run directory.

### Run TFOpt

1. Select `TFOpt local (tfopt-local)`.
2. Upload or select RNA/mRNA and TF-network inputs.
3. Pass a config file through the existing config argument when needed.
4. Review and run the generated command.
5. Inspect TFOpt tables, plots, logs, and workflow tabs after completion.

### Run ProtWise/model

1. Select `ProtWise model (protwise-model)`.
2. Assign the required protein/phosphosite inputs and optional config file.
3. Preview the command and run it.
4. Use the ProtWise panel to inspect predictions, residuals, fits, sensitivity files, plots, and reports.

### Run Networkmodel

1. Select `Networkmodel (networkmodel)`.
2. Assign kinase-network, TF-network, omics, and config inputs supported by the CLI.
3. Review the structured command preview.
4. Run the workflow.
5. Inspect scalar objective, convergence, predictions, optimization/profile/posterior outputs, plots, and the dashboard bundle when available.

### Download a result ZIP

1. Open a result directory in **Browse results**.
2. Open the **Download** tab.
3. Click **Prepare ZIP archive**.
4. Click **Download result ZIP**.

The ZIP is generated in memory and is not committed to the repository.

## Troubleshooting

| Symptom | Suggested action |
| --- | --- |
| Dashboard command is unavailable | Confirm Pixi is installed and run commands from the repository root. |
| Result directory is missing | Check the selected folder path and whether the workflow wrote to the expected `results/<workflow>/<run_id>/` directory. |
| `metadata.json` is missing or malformed | The browser can still show files, but provenance is incomplete. Re-run through an updated CLI wrapper when reproducibility is required. |
| CSV/TSV/XLSX preview fails | Verify the file is not empty, is readable, and matches its extension. |
| Workflow fails | Review the return code and `console.log` tail shown by the launcher. Copy the command preview into a terminal for deeper debugging if needed. |
| Optional visualization is unavailable | Install the optional Pixi environment or dependency for that visualization; core browsing remains available. |
| Large HTML/report previews are truncated | Download the original file from the browser or result ZIP. |
