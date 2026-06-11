# Result directory contract

PhosKinTime workflows should write outputs into a predictable result directory so the dashboard, tests, and downstream tools can parse runs without workflow-specific guesses.

## Standard layout

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

`config_resolved.toml` is also acceptable when a workflow naturally writes TOML. Current shared helpers write `config_resolved.yaml`.

## Files and folders

| Path | Required? | Purpose | Expected formats | Produced by | Dashboard use |
| --- | --- | --- | --- | --- | --- |
| `metadata.json` | Required for new runs | Machine-readable provenance: workflow, environment, config, inputs, status, settings. | JSON | CLI wrappers, dashboard runner, backend runners | Rendered in Metadata tab; used to infer workflow. |
| `command.txt` | Required for new runs | Exact or reconstructed invocation. | Plain text | CLI wrappers and dashboard runner | Rendered as command provenance. |
| `console.log` | Required when run through CLI/dashboard logging | Combined stdout/stderr or logger output. | Text/log | CLI wrappers and dashboard runner | Rendered in Logs tab and failure tail. |
| `config_resolved.yaml` / `config_resolved.toml` | Required when config/defaults are used | Effective config after default resolution and CLI overrides. | YAML/TOML/JSON-compatible YAML | Config-aware workflows | Rendered in Metadata tab for reproducibility. |
| `tables/` | Required folder, contents optional | Tabular outputs and mirrored legacy tables. | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.npy`, `.npz`, `.parquet` | All workflows as applicable | Listed lazily; shown with `st.dataframe` where supported. |
| `plots/` | Required folder, contents optional | Static and interactive figures. | `.png`, `.jpg`, `.jpeg`, `.svg`, `.html`, sometimes `.pdf` | Optimization, fitting, analysis workflows | Displayed with image/HTML viewers. |
| `logs/` | Required folder, contents optional | Auxiliary logs beyond root `console.log`. | Text/log files | Workflows and analysis scripts | Listed in Logs tab. |
| `reports/` | Required folder, contents optional | Human-readable reports. | `.html`, `.md`, `.pdf` | ProtWise, Networkmodel, analysis scripts | Rendered when possible; downloadable. |
| `artifacts/` | Required folder, contents optional | Serialized states and non-table artifacts. | `.pkl`, `.pickle`, `.joblib`, arbitrary files | Networkmodel, analyses, dashboard bundles | Downloadable; some workflow panels may parse known artifacts. |

## Metadata schema

New producers should include at least the following conceptual fields. Names may differ slightly for legacy runs, but dashboard-facing writers should prefer these stable keys when possible.

```json
{
  "workflow": "networkmodel",
  "run_id": "example",
  "timestamp_start": "...",
  "timestamp_end": "...",
  "status": "success",
  "return_code": 0,
  "command": ["pixi", "run", "..."],
  "config_path": "config.toml",
  "custom_config_used": false,
  "output_dir": "results/example",
  "python_version": "...",
  "package_version": "...",
  "git_commit": "...",
  "inputs": {},
  "parameters": {}
}
```

Current shared metadata helpers also record fields such as `command_arguments`, `output_directory`, `python_executable`, `pixi_environment`, input hashes, and workflow-specific effective settings.

Recommended metadata details:

- Use paths relative to the repository root where possible.
- Include resolved absolute paths only when useful for debugging local runs.
- Include file hashes for user inputs where practical.
- Record config priority outcomes: supplied `--conf`, resolved config path, default/custom config source, and effective settings.
- Record status and return code for dashboard-launched subprocesses.

## Config precedence

Workflow runners should resolve settings in this order:

```text
explicit CLI/dashboard field
> custom --conf
> repository config.toml
> hard-coded fallback only where unavoidable
```

`config_resolved.*` should represent the effective configuration after this precedence is applied.

## Legacy layout support

The dashboard result parser recognizes selected legacy outputs so older runs can still be browsed. Known examples include:

- Networkmodel top-level `scalar_objective.csv`, `convergence_history.csv`, `pred_prot_picked.csv`, `pred_rna_picked.csv`, `pred_phospho_picked.csv`.
- Networkmodel `optimization/`, `profiles/`, `posterior/`, and `plots/` folders.
- Legacy `kinopt_results.xlsx` and `tfopt_results.xlsx` workbooks.
- Top-level `report.html` where present.

Legacy support is compatibility behavior. New workflow code should still write the standard folders and provenance files.

Integrated `phoskintime-all` runs may contain child workflow result folders such as `tfopt/`, `kinopt/`, and `protwise/` beneath the parent run directory. The parent directory records launcher-level provenance, while each child directory remains independently browseable with its own tables, plots, reports, logs, artifacts, and provenance where produced.

## Workflow-specific expected outputs

### KinOpt

Expected outputs include:

- `kinopt_results.xlsx` or a mirrored copy under `tables/`;
- alpha/beta or equivalent parameter tables;
- observed-vs-estimated sheets/tables when available;
- fit and diagnostic plots under `plots/`;
- logs and provenance files.

### TFOpt

Expected outputs include:

- `tfopt_results.xlsx` or a mirrored copy under `tables/`;
- TF alpha/beta or equivalent tables;
- latent activity, dominance, knockout, or fit outputs when generated;
- plots and reports under standard folders;
- logs and provenance files.

### ProtWise

Expected outputs include:

- protein-wise ODE fit workbook/tables;
- residuals, fitted predictions, sensitivity outputs when generated;
- goodness-of-fit and diagnostic plots;
- `report.html` under `reports/` or mirrored from top level;
- provenance and resolved config.

### Networkmodel

Expected outputs include:

- scalar objective/convergence CSVs;
- picked protein/RNA/phospho prediction CSVs;
- `dashboard_bundle.pkl` and optimization artifacts;
- optimization/profile/posterior subfolders;
- fit plots and steady-state/network plots;
- provenance and resolved config.

### Scripts and advanced analyses

Advanced analysis scripts should write into the selected run directory:

- tables to `tables/`;
- figures to `plots/`;
- narrative outputs to `reports/`;
- serialized or miscellaneous outputs to `artifacts/`;
- logs to `logs/` or root `console.log` when run through the dashboard runner.

Scripts should not write into input data folders unless an explicit, documented export action requests it.
