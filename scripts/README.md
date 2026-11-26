# scripts/

These are stand-alone scripts that sit on top of the library packages (`protwise`, `networkmodel`, …). They are not part of the library: nothing in the library imports them. The dashboard calls the `analysis/` scripts.

| Folder | Contents |
|---|---|
| [`analysis/`](analysis/README.md) | Analysis backends used by the dashboard's "Advanced analysis" panel. Each takes CLI arguments. |
| [`export/`](export/README.md) | Model export: SBML (reaction form) and model documentation. |
| [`figures/egfr_cbc2026/`](figures/egfr_cbc2026/README.md) | EGFR inference, ensembles and CBC 2026 poster figures. They are kept as provenance. |
| [`apps/`](apps/README.md) | Stand-alone Streamlit viewers and the launcher for the legacy networkmodel dashboard. |
| [`legacy/`](legacy/README.md) | Older interactive tools kept for reference. |

## How to run

Run every script through pixi from the repository root. Pixi activation puts the repository root on `PYTHONPATH` (`[activation.env]` in `pixi.toml`), so the library packages can be imported:

```bash
pixi run python scripts/<folder>/<script>.py --help
```

Default input and output paths are relative to the current working directory, so running from the repository root matters. Several scripts read `data/…` (git-ignored) or a networkmodel results folder (`results_*`), and none of those are in the repository.

Apart from the two dynestyx scripts in `figures/egfr_cbc2026/`, which deliberately import `networkmodel` from the checkout given by `--source`, the scripts make no `sys.path` changes of their own. Outside pixi (`pixi run`, `pixi shell`, or the `.envrc` direnv hook), set `PYTHONPATH` to the repository root yourself.

Each folder's README lists, per script, the inputs, the default outputs, and how to run it.
