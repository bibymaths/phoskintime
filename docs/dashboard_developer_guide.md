# Dashboard developer guide

## Architecture

The dashboard package is organized as a thin UI and orchestration layer around existing backend CLIs.

```text
dashboard/
├── app.py
├── command_builder.py
├── runner.py
├── result_parser.py
├── file_utils.py
├── registry.py
├── components/
└── workflow_panels/
```

Key responsibilities:

- `app.py` wires the Streamlit tabs together.
- `registry.py` describes workflows, CLI flags, input assignment roles, and file-extension rules.
- `command_builder.py` converts structured selections into argv lists.
- `runner.py` executes subprocesses from the repository root and streams logs.
- `result_parser.py` lazily inventories standard and legacy result layouts.
- `file_utils.py` handles safe path, upload, preview, and ZIP helpers.
- `components/` contains reusable Streamlit render functions.
- `workflow_panels/` contains workflow-specific result viewers and analysis command wrappers.

## Design principles

- Do not put scientific computation in Streamlit UI files.
- Do not execute user-generated shell strings.
- Never use `shell=True` for dashboard-built workflow commands.
- Build commands as `list[str]` argv values.
- Backend CLI remains the source of truth.
- Result parsing must support both the standard result directory contract and practical legacy layouts.
- Optional dependencies must fail gracefully and only be imported on the render path that needs them.
- Importing dashboard modules should not start Streamlit.
- File validation must match backend readers.

## Workflow registry

`dashboard/registry.py` defines `WorkflowDescriptor`, `ArgumentSpec`, and `InputSpec` records.

Registry rules:

- Accepted file extensions must match backend readers.
- Required inputs must match CLI behavior.
- Labels/help text must not overpromise support.
- `--conf` must be forwarded and honored by the backend runner.
- Explicit CLI flags override config values.
- `output_dir_arg` must match the workflow CLI (`--outdir` or `--output-dir`).
- `safe_for_dashboard` should be `False` for result-only descriptors or unsafe workflows.

Important file-format examples:

- ProtWise protein input is CSV because the runner reads it with `pd.read_csv`.
- Networkmodel network/data inputs are CSV because `networkmodel.io.load_data` uses default `pd.read_csv`.
- Previous KinOpt/TFOpt prior result inputs are Excel workbooks where the backend uses Excel readers.

## Command builder

`dashboard/command_builder.py` resolves a workflow key, structured arguments, input assignments, Pixi environment, and run name into a `BuiltCommand`.

Precedence for configuration and arguments should remain:

```text
explicit dashboard/CLI selection
> uploaded/selected config
> repository default config
> hard-coded fallback only where unavoidable
```

Guidelines:

- Reject unknown argument names.
- Reject unknown input roles.
- Sanitize run names before constructing output paths.
- Keep output directories under the project `results/` area unless the user explicitly selects another path.
- Keep the command preview reproducible.

## Runner

`dashboard/runner.py` runs command lists with `subprocess.Popen(..., shell=False)` from the repository root. It merges stdout/stderr, streams output to the dashboard, and writes `console.log` in the run directory.

Return status conventions:

- `success` when return code is `0`.
- `failure` when return code is non-zero.
- `cancelled` when cancellation is requested and the process is terminated.

Interactive cancellation is intentionally limited until a robust background job supervisor is introduced. The runner exposes a cancellation callback boundary for supervised execution.

## Result parser

`dashboard/result_parser.py` inventories files without eagerly loading large content. The parser recognizes:

- standard files: `metadata.json`, `command.txt`, `console.log`, `config_resolved.yaml`;
- standard folders: `tables/`, `plots/`, `logs/`, `reports/`, `artifacts/`;
- legacy Networkmodel outputs such as `scalar_objective.csv`, prediction CSVs, `optimization/`, `profiles/`, `posterior/`, and `plots/`;
- legacy local result workbooks such as `kinopt_results.xlsx` and `tfopt_results.xlsx`.

Load file contents only inside viewer components after the user selects a file.

## Workflow panels

Workflow panels should present existing outputs and call reusable backend functions where needed. Do not copy scientific logic into Streamlit callbacks.

Existing apps and scripts to consider when adding or improving panels:

- `app/kinopt.py`
- `app/tfopt.py`
- `networkmodel/dashboard_app.py`
- `scripts/compare_mechanisms.py`

Refactor shared logic into reusable backend functions rather than copying code into dashboard files.

## Adding a new workflow

```text
1. Add or verify backend CLI
2. Ensure --outdir support
3. Ensure metadata/command/log output
4. Add registry entry
5. Add input specs
6. Add command builder tests
7. Add result parser tests
8. Add docs
```

Additional checklist:

- Verify `--conf` precedence if the workflow supports config files.
- Add result-panel discovery helpers for workflow-specific outputs.
- Add missing-file tests.
- Add file-format validation tests matching backend readers.
- Ensure generated outputs are ignored by Git.

## Testing

Run the full configured dev test task:

```bash
pixi run -e dev test
```

Useful focused tests:

```text
tests/dashboard/test_command_builder.py
tests/dashboard/test_result_parser.py
tests/dashboard/test_registry.py
tests/dashboard/test_file_utils.py
tests/dashboard/test_runner.py
tests/dashboard/test_upload_validation.py
tests/dashboard/test_workflow_panels.py
tests/test_result_contract.py
tests/test_networkmodel_runner_config.py
tests/test_protwise_config_handling.py
```

Docs build:

```bash
pixi run -e docs docs-build
```

## Git hygiene

Generated/local folders should remain ignored and uncommitted:

```text
dashboard_uploads/
results/
logs/
*.zip
.streamlit/
__pycache__/
```

Do not commit result folders, uploaded files, logs, ZIP archives, local Streamlit state, or cache directories.
