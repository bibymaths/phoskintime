"""File contract for result workbooks produced by the external KinOpt and TFOpt tools.

PhosKinTime does not import, install or vendor KinOpt or TFOpt. It reads their result
workbooks, validates them against the contract below, and records their provenance.

Contract (sheet -> required columns), derived from what PhosKinTime actually reads:

``kinopt_results.xlsx`` (KinOpt)
    "Alpha Values": Gene, Psite, Kinase, Alpha
        networkmodel/io.py: kinase->site edge weights (merged on Gene/Psite/Kinase).
    "Beta Values":  Kinase, Psite, Beta
        networkmodel/io.py: kinase priors; rows with an empty Psite are the kinase-level values.
    "Estimated":    Gene, Psite (+ time columns x1..x14)
        protwise/runner/main.py: phosphosite time courses (config [ode.inputs] psite_excel).
        protwise asks this module only for the sheet's presence and checks the columns itself.

``tfopt_results.xlsx`` (TFOpt)
    "Alpha Values": mRNA, TF, Value
        networkmodel/io.py: TF->target edge weights (merged on mRNA/TF).
    "Beta Values":  TF, PSite, Value
        networkmodel/io.py: TF priors; rows with an empty PSite are the TF-level values.
    "Estimated":    mRNA (+ time columns x1..x9)
        protwise/runner/main.py: mRNA time courses (config [ode.inputs] rna_excel); columns
        checked by protwise itself, as above.

Each consumer validates only the sheets it reads. Extra sheets and extra columns are allowed.

Optional sheet "Metadata" (any capitalisation): two columns, key and value, e.g. rows
``tool``, ``version``, ``git_commit``. If present, its contents are returned and recorded
in run provenance together with the file's SHA-256.

Errors
    ExternalResultsMissing (a FileNotFoundError): no path configured, or the file does not exist.
    ExternalResultsError (a ValueError): the file exists but cannot be read as a workbook,
        or a required sheet or column is missing. The message names the file, the sheet and
        the missing columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from common.results import file_sha256

DOCS_HINT = ("See the 'External tools' page of the PhosKinTime documentation "
             "(docs/Documentation/external_tools.md) for the file contract.")


@dataclass(frozen=True)
class ToolContract:
    """Required sheets and columns of one external tool's result workbook."""

    tool: str
    default_filename: str
    sheets: Mapping[str, tuple[str, ...]]


KINOPT = ToolContract(
    tool="KinOpt",
    default_filename="kinopt_results.xlsx",
    sheets={
        "Alpha Values": ("Gene", "Psite", "Kinase", "Alpha"),
        "Beta Values": ("Kinase", "Psite", "Beta"),
        "Estimated": ("Gene", "Psite"),
    },
)

TFOPT = ToolContract(
    tool="TFOpt",
    default_filename="tfopt_results.xlsx",
    sheets={
        "Alpha Values": ("mRNA", "TF", "Value"),
        "Beta Values": ("TF", "PSite", "Value"),
        "Estimated": ("mRNA",),
    },
)


class ExternalResultsMissing(FileNotFoundError):
    """The configured external result file is not set or does not exist."""


class ExternalResultsError(ValueError):
    """The external result file exists but does not satisfy the contract."""


@dataclass
class ExternalResults:
    """A validated external result workbook."""

    tool: str
    path: Path
    sha256: str | None
    sheets: dict[str, pd.DataFrame]
    metadata: dict[str, Any] | None = None

    def provenance(self, role: str) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "role": role,
            "status": "loaded",
            "path": str(self.path),
            "resolved_path": str(self.path.resolve()),
            "sha256": self.sha256,
            "metadata": self.metadata,
            "sheets_read": sorted(self.sheets),
        }


def absent_provenance(contract: ToolContract, path: Any, role: str) -> dict[str, Any]:
    """Provenance record for a result file that was not used because it is absent."""
    return {
        "tool": contract.tool,
        "role": role,
        "status": "absent",
        "path": None if path in (None, "") else str(path),
        "priors": "none",
    }


def is_absent(path: Any) -> bool:
    """True if no path is configured or the configured file does not exist."""
    return path is None or str(path).strip() == "" or not Path(str(path)).expanduser().exists()


def _read_metadata_sheet(xls: pd.ExcelFile) -> dict[str, Any] | None:
    name = next((s for s in xls.sheet_names if str(s).strip().lower() == "metadata"), None)
    if name is None:
        return None
    df = pd.read_excel(xls, sheet_name=name, header=None)
    meta: dict[str, Any] = {}
    for _, row in df.iterrows():
        if len(row) < 2 or pd.isna(row.iloc[0]):
            continue
        key = str(row.iloc[0]).strip()
        if key.lower() in ("key", "field", "name"):  # header row
            continue
        value = row.iloc[1]
        meta[key] = None if pd.isna(value) else (value.item() if hasattr(value, "item") else value)
    return meta


def read_external_results(path: Any, contract: ToolContract, sheets: Iterable[str],
                          required_columns: Mapping[str, Iterable[str]] | None = None) -> ExternalResults:
    """Read and validate the given sheets of an external result workbook.

    Args:
        path: Path to the workbook.
        contract: KINOPT or TFOPT.
        sheets: The sheets the caller will use. Each must be part of the contract.
        required_columns: Optional per-sheet override of the contract columns (subset checks
            for callers that validate further columns themselves).

    Raises:
        ExternalResultsMissing: no path configured, or the file does not exist.
        ExternalResultsError: unreadable workbook, or a missing sheet or column.
    """
    sheets = list(sheets)
    unknown = [s for s in sheets if s not in contract.sheets]
    if unknown:
        raise KeyError(f"{contract.tool} contract has no sheet(s) {unknown}")
    if is_absent(path):
        raise ExternalResultsMissing(
            f"{contract.tool} result file not found: {path!r}. Expected a workbook such as "
            f"'{contract.default_filename}'. {DOCS_HINT}")

    p = Path(str(path)).expanduser()
    try:
        xls = pd.ExcelFile(p)
    except Exception as exc:  # unreadable or not an Excel workbook
        raise ExternalResultsError(
            f"{contract.tool} result file {p} could not be read as an Excel workbook: {exc}. {DOCS_HINT}") from exc

    with xls:
        missing_sheets = [s for s in sheets if s not in xls.sheet_names]
        if missing_sheets:
            raise ExternalResultsError(
                f"{contract.tool} result file {p} is missing required sheet(s) {missing_sheets}; "
                f"found sheets {list(xls.sheet_names)}. {DOCS_HINT}")

        frames: dict[str, pd.DataFrame] = {}
        for sheet in sheets:
            df = pd.read_excel(xls, sheet_name=sheet)
            needed = tuple(required_columns.get(sheet, contract.sheets[sheet])) if required_columns \
                else contract.sheets[sheet]
            missing_cols = [c for c in needed if c not in df.columns]
            if missing_cols:
                raise ExternalResultsError(
                    f"{contract.tool} result file {p}, sheet '{sheet}': missing required column(s) "
                    f"{missing_cols}; found columns {list(map(str, df.columns))}. {DOCS_HINT}")
            frames[sheet] = df
        metadata = _read_metadata_sheet(xls)

    return ExternalResults(tool=contract.tool, path=p, sha256=file_sha256(p), sheets=frames, metadata=metadata)
