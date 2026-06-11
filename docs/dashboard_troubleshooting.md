# Dashboard troubleshooting

## Quick diagnostic table

| Symptom | Likely cause | Fix | Diagnostic command |
| --- | --- | --- | --- |
| Dashboard does not start | Dependencies missing or Pixi environment incomplete | Reinstall and run dashboard from repo root | `pixi install && pixi run dashboard` |
| Streamlit import fails | Streamlit missing from active env | Use Pixi environment, not system Python | `pixi run python -c "import streamlit; print(streamlit.__version__)"` |
| Pixi command fails | Pixi not installed or environment broken | Reinstall Pixi/env | `pixi run python --version` |
| Uploaded file rejected | Registry extension validation failed | Use the backend-supported extension | Check dashboard validation panel |
| Custom `--conf` appears ignored | Runner imported config constants too early or CLI field overrides config | Inspect provenance | `cat results/<run_id>/metadata.json` |
| Result folder is empty | Workflow failed, wrong `--outdir`, permissions, or wrong selected folder | Inspect logs and command | `cat results/<run_id>/console.log` |
| Tables/plots are missing | Legacy output layout or workflow wrote elsewhere | Check metadata/command output path | `cat results/<run_id>/command.txt` |
| UI freezes during long run | Heavy workflow running in Streamlit callback/process | Use subprocess workflow launcher; wait for logs | Inspect `console.log` |
| Optional visualization missing | Optional package not installed | Add dependency with Pixi | `pixi add plotly` |
| Tests fail after dashboard edits | Registry/command/result contract changed | Run focused tests | `pixi run -e dev pytest tests/dashboard -v` |

## Dashboard does not start

Install or refresh dependencies:

```bash
pixi install
pixi run dashboard
```

Check Streamlit:

```bash
pixi run python -c "import streamlit; print(streamlit.__version__)"
```

Run from the repository root. Streamlit should print a local URL such as `http://localhost:8501`.

## Pixi environment is broken

Basic checks:

```bash
pixi install
pixi run python --version
pixi run -e dev test
```

If the environment is corrupt, remove and recreate it:

```bash
rm -rf .pixi/envs/default
pixi install
```

## PyCharm notebook/kernel issues

PyCharm may use the wrong interpreter or Jupyter kernel. Check available kernels and the executable:

```bash
pixi run jupyter kernelspec list
pixi run python -c "import sys; print(sys.executable)"
```

The expected interpreter is inside the Pixi environment:

```text
.pixi/envs/default/bin/python
```

Register a kernel for notebooks:

```bash
pixi run python -m ipykernel install --user --name phoskintime --display-name "Python (phoskintime)"
```

Select `Python (phoskintime)` in PyCharm/Jupyter.

## `pip._internal.operations.build` error

This usually means the Pixi environment's `pip` is corrupted, or PyCharm is trying to install packages through its own helper instead of using Pixi.

Fix:

```bash
rm -rf .pixi/envs/default
pixi install
pixi add pip ipykernel
```

Do not rely on the PyCharm package installer for Pixi-managed dependencies. Add dependencies through `pixi.toml`/`pixi add` so the environment remains reproducible.

## Uploaded file rejected

The upload panel accepts a broad set of extensions, but workflow assignment validation follows backend readers.

- Config files assigned to workflow `--conf` fields currently must be TOML; YAML/JSON uploads are for preview/preset contexts unless a runner explicitly supports them.
- ProtWise protein input currently expects CSV when the backend uses `pd.read_csv`.
- Networkmodel kinase network, TF network, MS/protein, RNA, and phosphoproteomics inputs currently expect CSV with the default `pd.read_csv` reader.
- ProtWise phosphosite/RNA and previous KinOpt/TFOpt result inputs are Excel where the backend uses Excel readers.

Use the extension shown in the validation message.

## TSV/XLSX accepted but workflow fails

This should not happen after registry validation fixes. If it happens, the dashboard registry and backend reader are inconsistent.

Developer fix:

1. Check the backend reader (`pd.read_csv`, `pd.read_excel`, custom separator, etc.).
2. Update `dashboard/registry.py` `InputSpec.extensions` to match the reader.
3. Add a validation test in `tests/dashboard/test_upload_validation.py`.
4. Update user docs.

## Custom `--conf` appears ignored

Expected priority:

```text
explicit CLI/dashboard field
> custom --conf
> default config.toml
```

Diagnostics:

```bash
cat results/<run_id>/metadata.json
cat results/<run_id>/config_resolved.yaml
cat results/<run_id>/command.txt
```

Check these fields:

- supplied config path;
- resolved config path;
- config source/custom flag;
- effective inputs;
- effective bounds/lambdas/time grids;
- output directory.

If a workflow imports config constants at module load time, it may ignore custom `--conf`. This is a bug and should be fixed in the runner by parsing `--conf` before importing config-dependent modules.

## Result folder is empty

Likely causes:

- workflow failed before writing outputs;
- wrong `--outdir`/`--output-dir`;
- permission issue;
- dashboard pointed at the wrong result directory.

Diagnostics:

```bash
cat results/<run_id>/console.log
cat results/<run_id>/command.txt
```

If `console.log` is missing, the runner may have failed before result contract initialization.

## Dashboard cannot find tables/plots

Likely causes:

- legacy output layout;
- workflow did not follow result directory contract;
- output was written elsewhere;
- selected directory is a parent folder rather than the run folder.

Fix:

1. Open `metadata.json` and `command.txt`.
2. Confirm the command used the expected output directory.
3. Search the selected run folder for generated files.
4. If needed, update `dashboard/result_parser.py` to recognize a documented legacy layout.

## Long-running workflow freezes UI

Heavy workflows should run as subprocesses or external workers, not as scientific computation inside Streamlit callbacks. The dashboard launcher uses subprocess execution for registered workflows. If a new panel triggers expensive analysis directly, refactor it into a backend CLI/function and call it through the runner or a supervised job mechanism.

## Missing optional dependency

Examples:

- `gravis`
- `networkx`
- `imageio`
- `openpyxl`
- `plotly`

Install through Pixi only:

```bash
pixi add openpyxl
pixi add plotly
```

For visualization extras already modeled as a Pixi feature, prefer the appropriate Pixi environment if documented.

## Tests fail after dashboard changes

Run the configured test task:

```bash
pixi run -e dev test
```

Isolate dashboard tests:

```bash
pixi run -e dev pytest tests/dashboard -v
```

Common causes:

- registry `InputSpec` does not match backend reader;
- command builder accepts unsupported arguments;
- result parser no longer recognizes a legacy layout;
- a dashboard module imports Streamlit or optional heavy dependencies at module import time.
