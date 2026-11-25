# scripts/

These are stand-alone scripts that sit on top of the library packages (`kinopt`, `tfopt`, `protwise`, `networkmodel`, …). They are not part of the library: nothing in the library imports them. The dashboard calls the `analysis/` scripts.

| Folder | Contents |
|---|---|
| [`analysis/`](analysis/README.md) | Analysis backends used by the dashboard's "Advanced analysis" panel. Each takes CLI arguments. |
| [`export/`](export/README.md) | Model export: SBML (reaction form) and model documentation. |
| [`figures/egfr_cbc2026/`](figures/egfr_cbc2026/README.md) | EGFR inference, ensembles and CBC 2026 poster figures. They are kept as provenance. |
| [`figures/kinopt_tfopt/`](figures/kinopt_tfopt/README.md) | KinOpt/TFOpt network readouts, figures and schematic diagrams. |
| [`apps/`](apps/README.md) | Stand-alone Streamlit viewers and the launcher for the legacy networkmodel dashboard. |
| [`legacy/`](legacy/README.md) | Older interactive tools kept for reference. |

## How to run

Run every script from the repository root, inside the pixi environment. The repository root must be on `PYTHONPATH` so the library packages can be imported:

```bash
PYTHONPATH=. pixi run python scripts/<folder>/<script>.py --help
```

Default input and output paths are relative to the current working directory, so running from the repository root matters. Several scripts read `data/…` (git-ignored) or a networkmodel results folder (`results_*`), and none of those are in the repository.

Each folder's README lists, per script, the inputs, the default outputs, and how to run it.
