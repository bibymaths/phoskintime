# Dashboard quickstart

The PhosKinTime no-code dashboard is a Streamlit interface for users who want to run existing PhosKinTime workflows and inspect completed result folders without writing Python code.

## What the dashboard does

- Browses existing PhosKinTime result directories.
- Shows metadata, commands, logs, tables, plots, reports, artifacts, and ZIP downloads.
- Lets users upload/select input files into a dashboard-managed upload folder.
- Builds and previews safe CLI/Pixi commands for registered workflows.
- Runs registered workflows as subprocesses and streams console output.

## What the dashboard does not do

- It does not replace the CLI or Python API.
- It does not implement new scientific models.
- It does not silently convert file formats.
- It does not copy uploaded files into `data/` unless a user or backend command explicitly does so.
- It does not make long-running optimization jobs instant.

## Install dependencies

From the repository root:

```bash
pixi install
```

For development and test dependencies:

```bash
pixi run -e dev test
```

## Launch the dashboard

```bash
pixi run dashboard
```

Development mode with Streamlit reload-on-save:

```bash
pixi run dashboard-dev
```

After Streamlit starts, the terminal prints a local URL, usually `http://localhost:8501`. Open that URL in a browser. If Streamlit can open a browser automatically on your machine, it may do so.

## Browse existing results

1. Open the **Browse results** tab.
2. Select a base folder, commonly `results`.
3. Pick a result directory.
4. Inspect metadata, command, logs, tables, plots, reports, artifacts, and workflow-specific panels.
5. Use the **Download** tab to create a ZIP archive in memory.

## Run a workflow from the dashboard

1. Open the **Run workflow** tab.
2. Select a workflow.
3. Upload or select files.
4. Assign files to workflow input roles.
5. Review validation messages.
6. Preview the generated command.
7. Click **Run workflow**.
8. Watch console output and open the completed result directory.

## Upload input files

Uploaded files are saved under:

```text
dashboard_uploads/<run_id>/
```

Supported upload extensions are `.csv`, `.tsv`, `.xlsx`, `.yaml`, `.yml`, `.json`, and `.txt`. Workflow execution validation may be stricter than upload validation. For example, Networkmodel execution inputs and the ProtWise protein input currently use CSV readers in the backend, so those assigned inputs must be `.csv`.

## Download results

The result browser can package the selected result directory into a ZIP file. The ZIP is generated in memory for browser download and should not be committed to Git.

## Minimal command examples

```bash
pixi install
pixi run dashboard
pixi run dashboard-dev
pixi run -e dev test
```

Workflow examples:

```bash
pixi run kinopt-local --outdir results/example_kinopt
pixi run tfopt-local --outdir results/example_tfopt
pixi run model --conf config.toml --outdir results/example_protwise
pixi run networkmodel --conf config.toml --output-dir results/example_networkmodel
```

## Checklist

```text
1. Start dashboard
2. Select or upload files
3. Select workflow
4. Preview command
5. Run
6. Inspect console
7. Browse results
8. Download ZIP
```
