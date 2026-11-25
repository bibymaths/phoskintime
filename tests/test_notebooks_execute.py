from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

# Paths are anchored to the repository root so the test does not depend on the
# working directory or on the checkout folder's name.
REPO_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    REPO_ROOT / "notebooks/01_kinopt_educational_workflow.ipynb",
    REPO_ROOT / "notebooks/02_tfopt_educational_workflow.ipynb",
    REPO_ROOT / "notebooks/03_protwise_educational_workflow.ipynb",
    REPO_ROOT / "notebooks/04_networkmodel_educational_workflow.ipynb",
]


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_educational_notebooks_execute(notebook_path: Path):
    """Execute each educational workflow notebook from top to bottom."""
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(REPO_ROOT)}})
    client.execute()
