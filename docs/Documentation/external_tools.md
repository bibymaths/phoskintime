# External tools: KinOpt and TFOpt

KinOpt and TFOpt are standalone tools. PhosKinTime does not import, install or vendor
them. It only reads their **result workbooks** (Excel files), checks them against the file
contract on this page, and records where they came from.

| Tool | Repository | Status |
|---|---|---|
| KinOpt | <https://github.com/bibymaths/KinOpt> | Public. No release is tagged yet, so refer to a KinOpt commit hash. |
| TFOpt | — | Repository not yet public. |

The `kinopt/` and `tfopt/` code was last included in PhosKinTime at commit `6eb04fa`.
Use the tools' own repositories for current code; `git show 6eb04fa:kinopt/...` shows the
last copy that lived here.

This page describes only the file interface. It does not describe the tools' equations or
what their parameters mean; see each tool's own documentation for that. Known issues that
apply to KinOpt/TFOpt are listed in `KNOWN_ISSUES.md` in the repository root, under
"Applies to KinOpt/TFOpt".

## File contract

The contract is defined in `common/external_results.py` (`KINOPT` and `TFOPT`). It lists
only what PhosKinTime actually reads.

### KinOpt: `kinopt_results.xlsx`

| Sheet | Required columns | Read by | Used for |
|---|---|---|---|
| `Alpha Values` | `Gene`, `Psite`, `Kinase`, `Alpha` | `networkmodel/io.py` | Kinase→site edge weights (merged on Gene/Psite/Kinase). |
| `Beta Values` | `Kinase`, `Psite`, `Beta` | `networkmodel/io.py` | Kinase priors. Rows with an empty `Psite` are the kinase-level values. |
| `Estimated` | `Gene`, `Psite` (+ time columns `x1`..`x14`) | `protwise/runner/main.py` | Phosphosite time courses. |

### TFOpt: `tfopt_results.xlsx`

| Sheet | Required columns | Read by | Used for |
|---|---|---|---|
| `Alpha Values` | `mRNA`, `TF`, `Value` | `networkmodel/io.py` | TF→target edge weights (merged on mRNA/TF). |
| `Beta Values` | `TF`, `PSite`, `Value` | `networkmodel/io.py` | TF priors. Rows with an empty `PSite` are the TF-level values. |
| `Estimated` | `mRNA` (+ time columns `x1`..`x9`) | `protwise/runner/main.py` | mRNA time courses. |

Notes:

- Each consumer checks only the sheets it reads. networkmodel checks `Alpha Values` and
  `Beta Values`; protwise checks only that `Estimated` is present and then checks its
  columns itself.
- Extra sheets and extra columns are allowed. KinOpt and TFOpt usually also write sheets
  such as `Observed` and `Residuals`; the contract ignores them.
- Column names are case-sensitive. Note `Psite` (KinOpt) versus `PSite` (TFOpt).

### Optional `Metadata` sheet

A sheet named `Metadata` (any capitalisation) with two columns, key and value, for
example:

| key | value |
|---|---|
| `tool` | `KinOpt` |
| `version` | `…` |
| `git_commit` | `…` |

If the sheet is present, its contents are recorded in the run's provenance. Since KinOpt
has no tagged release yet, `git_commit` is the most useful entry.

### Provenance

For every workbook it reads, PhosKinTime records in the run's `metadata.json`, under
`external_results`: the tool, the role of the file, the path as given and the resolved
path, the file's SHA-256, the sheets read, and the `Metadata` contents (if any). When
networkmodel runs without a KinOpt/TFOpt file, it records the file as absent with
`priors: none`.

## Where the paths are configured

All in `config.toml`. Relative paths are resolved against the repository root.

| Key | Tool | Read by | Default |
|---|---|---|---|
| `[networkmodel] kinopt` | KinOpt | networkmodel (priors) | `data/kinopt_results.xlsx` |
| `[networkmodel] tfopt` | TFOpt | networkmodel (priors) | `data/tfopt_results.xlsx` |
| `[ode.inputs] psite_excel` | KinOpt | protwise (`Estimated`) | `""` (must be set) |
| `[ode.inputs] rna_excel` | TFOpt | protwise (`Estimated`) | `""` (must be set) |

The same paths can be given on the command line: `--kinopt` / `--tfopt` for networkmodel,
`--input-excel-psite` / `--input-excel-rna` for protwise.

What happens when a file is missing differs between the two consumers:

- **networkmodel:** the KinOpt/TFOpt files are optional. If one is not configured or does
  not exist, a WARNING is logged, all edges of that kind get alpha = 1.0, no priors are
  used, and the provenance says `priors: none`. A file that exists but breaks the contract
  is an error.
- **protwise:** both files are required. A missing or malformed file is an error.
- **`pixi run phoskintime all`** (prep → check → model) checks both `[ode.inputs]` files
  after preprocessing and stops with exit code 2 before the model stage if one is missing
  or malformed. It does not run KinOpt or TFOpt.

## Errors

| Error | Base class | Meaning |
|---|---|---|
| `ExternalResultsMissing` | `FileNotFoundError` | No path is configured, or the file does not exist. |
| `ExternalResultsError` | `ValueError` | The file exists but cannot be read as an Excel workbook, or a required sheet or column is missing. The message names the file, the sheet and the missing columns. |

Both messages point back to this page.

## Other places that read the workbooks

- **Dashboard:** the KinOpt and TFOpt result-viewer panels
  (`dashboard/workflow_panels/kinopt.py`, `dashboard/workflow_panels/tfopt.py`) show an
  existing workbook. They do not run the tools. Contract problems are shown as warnings
  instead of errors, and the file's SHA-256 and `Metadata` sheet are displayed.
- **`common/utils/tables.py`:** takes a KinOpt workbook path and writes per-site
  LaTeX/CSV tables from `Alpha Values` and `Beta Values`
  (`pixi run python common/utils/tables.py <kinopt_results.xlsx>`).
- **`scripts/analysis/analyze_tf_kin_counts.py`:** reads `Alpha Values` and
  `Beta Values` from both workbooks (`--kinopt-xlsx`, `--tfopt-xlsx`).
- **`scripts/analysis/curve_similarity.py`:** reads `Observed` and `Estimated` from both
  workbooks (`--kinopt-xlsx`, `--tfopt-xlsx`). `Observed` is not part of the contract,
  so this script needs it in addition.

The two scripts read the sheets directly with pandas; they do not go through the contract
check.
