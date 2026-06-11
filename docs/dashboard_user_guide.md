# Dashboard user guide

## Overview

The PhosKinTime dashboard is a no-code wrapper around existing PhosKinTime command-line workflows. It helps users choose files, preview commands, launch supported workflows, monitor logs, and browse output folders.

```text
Dashboard = wrapper + launcher + result viewer
CLI = source of truth
Scientific modules = model logic
```

The dashboard should be used for convenience and reproducibility. The CLI and backend modules remain authoritative for scientific behavior.

## Supported workflows

| Workflow | Purpose | Expected inputs | Accepted execution formats | Output location | Runtime class | Dashboard execution | Result-only viewing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prep` | Run preprocessing cleanup via `processing.cleanup`. | Files referenced by repository configuration and preprocessing conventions. | Config/data dependent. | Existing preprocessing outputs; no dedicated dashboard output flag. | Short | Yes | Limited |
| `kinopt-local` / `kinopt` | Local kinase-to-phosphosite optimization. | Protein/kinase abundance CSV, phosphosite CSV/network data, optional config. | Registered config files: `.toml`, `.yaml`, `.yml`, `.json`; workflow data follows backend config. | `results/kinopt-local/<run_id>/` when launched by dashboard; CLI `--outdir` supported. | Medium | Yes | Yes |
| `kinopt-evol` | Evolutionary KinOpt mode exposed as a Pixi task. | Same family of KinOpt inputs, governed by backend config. | Backend dependent. | Backend-defined outputs. | Medium/long | Not directly registered in dashboard launcher | Result folders can be browsed if contract/legacy layout is present |
| `tfopt-local` / `tfopt` | Local TF-to-mRNA optimization. | mRNA/RNA CSV, TF network CSV, optional config. | Registered config files: `.toml`, `.yaml`, `.yml`, `.json`; workflow data follows backend config. | `results/tfopt-local/<run_id>/` when launched by dashboard; CLI `--outdir` supported. | Medium | Yes | Yes |
| `tfopt-evol` | Evolutionary TFOpt mode exposed as a Pixi task. | Same family of TFOpt inputs, governed by backend config. | Backend dependent. | Backend-defined outputs. | Medium/long | Not directly registered in dashboard launcher | Result folders can be browsed if contract/legacy layout is present |
| `protwise-model` / `model` | Protein-wise ODE fitting downstream of KinOpt/TFOpt. | Protein CSV, phosphosite Excel workbook, RNA/TFOpt Excel workbook, optional TOML config. | Protein input: `.csv`; phosphosite and RNA result inputs: `.xlsx`; config: `.toml`, `.yaml`, `.yml`, `.json` in dashboard assignment. | `results/protwise-model/<run_id>/` when launched by dashboard; CLI `--outdir` supported. | Medium/long | Yes | Yes |
| `networkmodel` | Coupled kinase-signaling and TF/RNA global network model. | Kinase network CSV, TF network CSV, MS/protein CSV, RNA CSV, optional phosphoproteomics CSV, optional KinOpt/TFOpt Excel priors, TOML config. | Network/data inputs read by backend with `pandas.read_csv`: `.csv` only; prior result workbooks: `.xlsx`. | `results/networkmodel/<run_id>/` when launched by dashboard; CLI `--output-dir`/`--outdir` supported. | Long | Yes | Yes |
| `phoskintime-all` | Typer wrapper for preprocessing, local TFOpt, local KinOpt, and ProtWise. | Stage-specific configs for TFOpt/KinOpt/ProtWise. | Config assignment: `.toml`, `.yaml`, `.yml`, `.json`. | `results/phoskintime-all/<run_id>/` when launched by dashboard. | Long | Yes | Stage result folders can be browsed |
| Advanced analysis scripts | Existing analysis utilities such as curve similarity, protein accumulator detection, subnetwork export, mechanistic insights, temporal sensitivity, and mechanism comparison. | Existing result folders/tables, depending on script. | Script-specific; do not assume upload conversion. | Selected run directory under `tables/`, `plots/`, `reports/`, or `artifacts/` when integrated. | Medium/long | Exposed as parameterized wrappers where available; not run automatically | Yes |

## Launching the dashboard

Install dependencies first:

```bash
pixi install
```

Start the user dashboard:

```bash
pixi run dashboard
```

Start development mode with reload-on-save:

```bash
pixi run dashboard-dev
```

Streamlit prints a local browser URL such as `http://localhost:8501`.

## Browsing existing results

Use **Browse results** to select a result directory. The dashboard discovers standard contract files and known legacy outputs.

The browser can show:

- `metadata.json` provenance.
- `command.txt` command provenance.
- `console.log` and `logs/*`.
- Tables from `tables/*.csv`, `tables/*.tsv`, and `tables/*.xlsx`.
- Plots from `plots/*.png`, `plots/*.jpg`, `plots/*.jpeg`, `plots/*.svg`, and `plots/*.html`.
- Reports from `reports/*.html`, `reports/*.md`, and PDFs where possible.
- Artifacts from `artifacts/*`.
- A downloadable ZIP archive of the selected result directory.

If standard files are missing, the dashboard still attempts to show recognized legacy outputs and reports what is missing.

## Uploading files

Uploaded files are stored in:

```text
dashboard_uploads/<run_id>/
```

Uploaded files are not silently copied into `data/`. The command preview points to the uploaded/selected path used by the backend.

General upload extensions accepted by the UI are:

```text
.csv, .tsv, .xlsx, .yaml, .yml, .json, .txt
```

Workflow execution validation is stricter and follows backend readers:

- ProtWise protein input currently expects CSV because the backend reads it with `pd.read_csv`.
- ProtWise phosphosite and RNA result inputs currently expect Excel workbooks.
- Networkmodel kinase network, TF network, MS/protein data, RNA data, and phosphoproteomics inputs currently expect CSV because the backend reads them with default `pd.read_csv`.
- Networkmodel previous KinOpt/TFOpt result priors remain Excel workbooks.

## Running workflows

The dashboard-generated command is shown before execution. These are representative commands using actual Pixi task names from `pixi.toml`:

```bash
pixi run kinopt-local --outdir results/example_kinopt
```

```bash
pixi run tfopt-local --outdir results/example_tfopt
```

```bash
pixi run model --conf config.toml --outdir results/example_protwise
```

```bash
pixi run networkmodel --conf config.toml --output-dir results/example_networkmodel
```

Additional actual tasks include:

```bash
pixi run prep
pixi run kinopt-evol
pixi run tfopt-evol
pixi run phoskintime-all
```

Use long-running workflows with care. Networkmodel and full integrated runs can take substantially longer than local parsing or result browsing.

## Command preview

The dashboard shows the exact command and argument list before running because:

- users can copy/paste it into a terminal;
- the command can be recorded in `command.txt`;
- it makes file paths and config choices visible;
- it prevents hidden free-text shell execution.

Commands are built as argument lists, not shell strings.

## Result interpretation

Common output groups:

| Folder | Meaning |
| --- | --- |
| `tables/` | CSV, TSV, Excel, JSON, NumPy, or other table-like outputs. |
| `plots/` | Static and interactive plots. |
| `logs/` | Auxiliary logs beyond the root `console.log`. |
| `reports/` | HTML, Markdown, or PDF reports. |
| `artifacts/` | Pickles, serialized bundles, intermediate arrays, and files that are useful for reloading or debugging. |

Workflow-specific tabs add convenience views for KinOpt, TFOpt, ProtWise, Networkmodel, and advanced analyses when recognized files are present.

## Downloading results

Open the **Download** tab in a selected result directory and click **Prepare ZIP archive**. The ZIP is generated in memory and served to the browser. It is not written into the repository by default.

## Reproducibility

A dashboard-ready run should contain:

- `metadata.json` — workflow identity, timestamps, environment details, config provenance, input hashes where practical, and selected parameters.
- `command.txt` — exact or reconstructed command invocation.
- `console.log` — stdout/stderr or logger output.
- `config_resolved.yaml` or `config_resolved.toml` — effective resolved configuration after defaults and CLI overrides.

Config precedence is:

```text
explicit CLI/dashboard field
> custom --conf
> repository config.toml
> hard-coded fallback only where unavoidable
```

## Known limitations

- Long-running jobs can take minutes or hours.
- Large result folders may load slowly, especially large HTML reports or many images.
- Optional visualization dependencies may be missing in minimal environments.
- The dashboard does not replace CLI/API workflows.
- The dashboard does not change scientific model assumptions.
- Workflow validation should match backend readers; if a backend cannot read TSV/XLSX, the dashboard should not permit those formats for execution.
