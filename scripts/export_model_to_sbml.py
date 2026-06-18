#!/usr/bin/env python
"""
Export fitted PhosKinTime results to SBML Level 3 Version 2.

This is intentionally read-only with respect to the input result directory.
It parses only confirmed fitted-parameter schemas and skips predictions,
trajectories, residuals, objectives, metrics, plots, summaries and pickles.

Networkmodel equations are grounded in networkmodel/backend.py:
- make_networkmodel_rhs
- multimodal_loss_from_trajectory
- _global_networkmodel_observable

Protwise equations are grounded in:
- protwise/models/diffrax_solver.py
- protwise/models/distmod.py, succmod.py, randmod.py when present

KinOpt/TFOpt workbook schemas are grounded in:
- kinopt/local/exporter/sheetutils.py::output_results
- tfopt/local/exporter/sheetutils.py::save_results_to_excel
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import libsbml  # type: ignore
except Exception:  # pragma: no cover
    libsbml = None  # type: ignore


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_id(text: Any, prefix: str = "id") -> str:
    """Return an SBML-safe SId while preserving recognisable source text."""
    s = str(text)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = prefix
    if not re.match(r"^[A-Za-z_]", s):
        s = f"{prefix}_{s}"
    return s


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def first_existing(root: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        for candidate in (root / name, root / "tables" / name):
            if candidate.is_file():
                return candidate
    return None


def ensure_libsbml() -> None:
    if libsbml is None:
        raise RuntimeError(
            "python-libsbml is required for SBML export. "
            "Install it with: pip install python-libsbml"
        )


def cell_compartment(model):
    comp = model.createCompartment()
    comp.setId("cell")
    comp.setName("dimensionless_cell")
    comp.setConstant(True)
    comp.setSpatialDimensions(0.0)
    comp.setSize(1.0)
    comp.setUnits("dimensionless")
    return comp


def add_notes(obj, paragraphs: Iterable[str]) -> None:
    body = '<body xmlns="http://www.w3.org/1999/xhtml">' + "".join(
        f"<p>{str(p)}</p>" for p in paragraphs
    ) + "</body>"
    obj.setNotes(body)


def create_parameter(model, pid: str, value: float, name: Optional[str] = None, constant: bool = True):
    pid = safe_id(pid, "p")
    if model.getParameter(pid) is not None:
        return model.getParameter(pid)
    p = model.createParameter()
    p.setId(pid)
    p.setName(name or pid)
    p.setValue(finite_float(value))
    p.setConstant(bool(constant))
    p.setUnits("dimensionless")
    return p


def create_species(model, sid: str, name: Optional[str] = None, initial: float = 0.0):
    sid = safe_id(sid, "s")
    if model.getSpecies(sid) is not None:
        return model.getSpecies(sid)
    sp = model.createSpecies()
    sp.setId(sid)
    sp.setName(name or sid)
    sp.setCompartment("cell")
    sp.setInitialAmount(finite_float(initial))
    sp.setSubstanceUnits("dimensionless")
    sp.setHasOnlySubstanceUnits(False)
    sp.setBoundaryCondition(False)
    sp.setConstant(False)
    return sp


def add_assignment_rule(model, variable: str, formula: str) -> None:
    r = model.createAssignmentRule()
    r.setVariable(safe_id(variable, "v"))
    r.setFormula(formula)


def add_rate_rule(model, variable: str, formula: str) -> None:
    r = model.createRateRule()
    r.setVariable(safe_id(variable, "v"))
    r.setFormula(formula)


def write_doc(doc, output_path: Path, validate: bool = False) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    libsbml.writeSBMLToFile(doc, str(output_path))
    summary = {"path": str(output_path), "errors": 0, "warnings": 0, "messages": []}
    if validate:
        n = doc.checkConsistency()
        for i in range(n):
            err = doc.getError(i)
            sev = int(err.getSeverity())
            msg = err.getMessage()
            if sev >= libsbml.LIBSBML_SEV_ERROR:
                summary["errors"] += 1
            else:
                summary["warnings"] += 1
            summary["messages"].append(msg)
    return summary


# ---------------------------------------------------------------------------
# Family inference and parsers
# ---------------------------------------------------------------------------

def infer_model_family(path: Path) -> Optional[str]:
    """Infer from contents. --model-family remains authoritative."""
    path = Path(path)
    names = {p.name for p in path.iterdir()} if path.exists() else set()
    if (path / "tables").is_dir():
        names |= {p.name for p in (path / "tables").iterdir()}
    if "kinopt_results.xlsx" in names:
        return "kinopt"
    if "tfopt_results.xlsx" in names:
        return "tfopt"
    if "model_parameters_genes.csv" in names or "network_W_global.csv" in names:
        return "networkmodel"
    if any(path.rglob("*_parameters.xlsx")) or (path / "Distributive" / "Distributive_results.xlsx").is_file():
        return "protwise"
    low = str(path).lower()
    for fam in ("networkmodel", "protwise", "kinopt", "tfopt"):
        if fam in low:
            return fam
    return None


def resolve_family(path: Path, requested: Optional[str]) -> tuple[str, str]:
    inferred = infer_model_family(path)
    if requested is not None:
        if inferred is not None and inferred != requested:
            # This is an explicit mismatch, not a fallback.
            raise ValueError(f"Requested --model-family={requested!r}, but directory looks like {inferred!r}.")
        return requested, "--model-family"
    if inferred is None:
        raise ValueError("Could not infer model family. Pass --model-family explicitly.")
    return inferred, "path"


def parse_kinopt_xlsx(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Schema source of truth:
    # - kinopt/local/exporter/sheetutils.py::output_results
    xls = pd.ExcelFile(path)
    sheets = {s.strip().lower(): s for s in xls.sheet_names}
    if "alpha values" not in sheets or "beta values" not in sheets:
        raise ValueError(f"{path} is missing KinOpt parameter sheets: Alpha Values and/or Beta Values")
    alpha = pd.read_excel(path, sheet_name=sheets["alpha values"])
    beta = pd.read_excel(path, sheet_name=sheets["beta values"])
    alpha_required = {"Gene", "Psite", "Kinase", "Alpha"}
    beta_required = {"Kinase", "Psite", "Beta"}
    if not alpha_required.issubset(alpha.columns):
        raise ValueError(f"{path}: Alpha Values sheet does not match KinOpt schema {alpha_required}")
    if not beta_required.issubset(beta.columns):
        raise ValueError(f"{path}: Beta Values sheet does not match KinOpt schema {beta_required}")
    return alpha, beta


def parse_tfopt_xlsx(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Schema source of truth:
    # - tfopt/local/exporter/sheetutils.py::save_results_to_excel
    xls = pd.ExcelFile(path)
    sheets = {s.strip().lower(): s for s in xls.sheet_names}
    if "alpha values" not in sheets or "beta values" not in sheets:
        raise ValueError(f"{path} is missing TFOpt parameter sheets: Alpha Values and/or Beta Values")
    alpha = pd.read_excel(path, sheet_name=sheets["alpha values"])
    beta = pd.read_excel(path, sheet_name=sheets["beta values"])
    alpha_required = {"mRNA", "TF", "Value"}
    beta_required = {"TF", "PSite", "Value"}
    if not alpha_required.issubset(alpha.columns):
        raise ValueError(f"{path}: Alpha Values sheet does not match TFOpt schema {alpha_required}")
    if not beta_required.issubset(beta.columns):
        raise ValueError(f"{path}: Beta Values sheet does not match TFOpt schema {beta_required}")
    return alpha, beta


def strict_parameter_csv(path: Path) -> pd.DataFrame:
    """Strict fallback for files explicitly named as alpha/beta/parameter tables."""
    stem = path.stem.lower()
    accepted = ("alpha", "beta", "parameters", "fitted_params", "estimated_params", "coefficients", "weights")
    rejected = ("trajectory", "residual", "metric", "score", "loss", "objective", "prediction", "pred_", "summary")
    if not any(k in stem for k in accepted) or any(k in stem for k in rejected):
        return pd.DataFrame()
    df = pd.read_csv(path)
    bad_cols = {"time", "replicate", "observed", "estimated", "predicted", "residual", "rmse", "mae", "r2", "score", "loss", "objective", "metric", "index", "id"}
    numeric = [c for c in df.columns if c.lower() not in bad_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return pd.DataFrame()
    return df[[c for c in df.columns if c in numeric or not pd.api.types.is_numeric_dtype(df[c])]]


GENE_COL_CANDIDATES = ("Protein_Gene", "gene", "Gene", "protein", "Protein", "protein_id", "ProteinID")
PSITE_COL_CANDIDATES = ("Psite", "psite", "site", "Site", "phosphosite", "Residue_Position")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def get_gene_column(df: pd.DataFrame) -> str:
    c = find_col(df, GENE_COL_CANDIDATES)
    if c is None:
        raise ValueError(f"Could not find a gene/protein column in {list(df.columns)}")
    return c


def get_psite_column(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, PSITE_COL_CANDIDATES)


def parse_networkmodel_tables(results_dir: Path) -> dict[str, Any]:
    # Schema source of truth:
    # - networkmodel/backend.py::_unpack_theta_jax
    # - networkmodel/backend.py::make_networkmodel_rhs
    parsed: dict[str, Any] = {"files_parsed": [], "files_skipped": [], "warnings": []}
    gene_path = results_dir / "model_parameters_genes.csv"
    psite_path = results_dir / "model_parameters_genes_psites.csv"
    kinase_path = results_dir / "model_parameters_kinases.csv"
    s_path = results_dir / "S_rates_picked.csv"
    if not gene_path.is_file():
        raise FileNotFoundError(f"Required networkmodel parameter file not found: {gene_path}")
    gene_df = pd.read_csv(gene_path)
    parsed["gene_df"] = gene_df
    parsed["gene_col"] = get_gene_column(gene_df)
    parsed["files_parsed"].append(gene_path)
    if psite_path.is_file():
        parsed["psite_df"] = pd.read_csv(psite_path)
        parsed["files_parsed"].append(psite_path)
    else:
        parsed["psite_df"] = pd.DataFrame()
    if kinase_path.is_file():
        parsed["kinase_df"] = pd.read_csv(kinase_path)
        parsed["files_parsed"].append(kinase_path)
    else:
        parsed["kinase_df"] = pd.DataFrame()
    if s_path.is_file():
        parsed["s_rates_df"] = pd.read_csv(s_path)
        parsed["files_parsed"].append(s_path)
    else:
        parsed["s_rates_df"] = pd.DataFrame()
    # structure/metadata only
    for name in [
        "initial_conditions_y0.csv", "optimized_entities.json", "metadata.json", "mode_metadata.json",
        "config_resolved.yaml", "network_W_global.csv", "network_kinase_inputs.csv", "network_tf_mat.csv",
    ]:
        p = results_dir / name
        if p.is_file():
            parsed["files_parsed"].append(p)
    # skipped non-parameter outputs
    skip_names = [
        "kinase_activities_dynamic.csv", "model_trajectories.csv", "pred_phospho_picked.csv",
        "pred_prot_picked.csv", "pred_rna_picked.csv", "residuals_table.csv", "scalar_objective.csv",
        "pareto_F.csv", "pareto_front.xlsx", "jaxopt_optimization_result.pkl",
    ]
    for name in skip_names:
        p = results_dir / name
        if p.is_file():
            parsed["files_skipped"].append(p)
    return parsed


def parse_protwise_parameters(results_dir: Path) -> dict[str, Any]:
    # Schema source of truth:
    # - protwise/paramest/core.py::process_gene
    # - common/utils/display.py::save_result
    parsed = {"gene_params": [], "files_parsed": [], "files_skipped": [], "latex_files": [], "warnings": []}
    for path in sorted(results_dir.rglob("*_parameters.xlsx")):
        if path.name.lower().endswith("_confidence_intervals.csv"):
            continue
        try:
            df = pd.read_excel(path)
        except Exception as exc:
            parsed["warnings"].append(f"Could not read {path}: {exc}")
            continue
        parameter_cols = [c for c in df.columns if str(c) not in {"Time", "Regularization"} and pd.api.types.is_numeric_dtype(df[c])]
        if not parameter_cols:
            parsed["warnings"].append(f"No numeric fitted parameter columns found in {path}")
            continue
        row = df[parameter_cols].iloc[-1]
        parsed["gene_params"].append({"gene": path.name.replace("_parameters.xlsx", ""), "path": path, "params": {c: finite_float(row[c]) for c in parameter_cols}})
        parsed["files_parsed"].append(path)
    for path in sorted(results_dir.rglob("*confidence_intervals.csv")):
        parsed["files_skipped"].append(path)
    for path in sorted(results_dir.rglob("*.png")) + sorted(results_dir.rglob("*.jpg")) + sorted(results_dir.rglob("*.pdf")):
        parsed["files_skipped"].append(path)
    for path in sorted(results_dir.rglob("*model_latex.tex")):
        parsed["latex_files"].append(path)
        parsed["files_parsed"].append(path)
    return parsed


def detect_protwise_model(params: dict[str, float]) -> str:
    names = list(params)
    d_like = [n for n in names if re.fullmatch(r"D\d+", str(n))]
    s_like = [n for n in names if re.fullmatch(r"S\d+", str(n))]
    combo_d = [n for n in names if re.fullmatch(r"D\d{2,}", str(n))]
    # D12 etc. are randmod combination dephosphorylation/decay terms.
    if combo_d:
        return "randmod"
    if len(d_like) == len(s_like):
        return "distmod"
    return "unknown"


# ---------------------------------------------------------------------------
# Networkmodel formula extraction
# ---------------------------------------------------------------------------

@dataclass
class NetworkGene:
    name: str
    A: float
    B: float
    C: float
    D: float
    E: float
    tf_scale: float
    sites: list[str] = field(default_factory=list)
    Dp: dict[str, float] = field(default_factory=dict)
    S: dict[str, float] = field(default_factory=dict)


PARAM_ALIASES = {
    "A": ("A_i", "Synthesis_A", "A", "mRNA_prod", "mRNA_production"),
    "B": ("B_i", "mRNA_Degradation_B", "B", "mRNA_deg", "mRNA_degradation"),
    "C": ("C_i", "Translation_C", "C", "protein_prod", "translation"),
    "D": ("D_i", "Protein_Degradation_D", "D", "protein_deg", "protein_degradation"),
    "E": ("E_i", "De-Phosphorylation_E", "De_Phosphorylation_E", "Dephosphorylation_E", "E"),
    "tf_scale": ("tf_scale", "Global_TF_Scale", "TF_Scale", "tf_scale_global"),
}


def row_value(row: pd.Series, aliases: Iterable[str], default: float = 0.0) -> float:
    for col in aliases:
        if col in row.index:
            return finite_float(row[col], default)
    lower = {str(c).lower(): c for c in row.index}
    for col in aliases:
        if col.lower() in lower:
            return finite_float(row[lower[col.lower()]], default)
    return default


def psite_value_column(df: pd.DataFrame) -> Optional[str]:
    candidates = (
        "Dp_i", "Phospho_Degradation_Dp", "Phospho_Degradation_Dp_value", "Dp",
        "D_p", "Ddeg", "D_rate", "Phospho_Degradation"
    )
    c = find_col(df, candidates)
    if c is not None:
        return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric:
        if any(k in str(c).lower() for k in ("dp", "degrad", "ddeg")):
            return c
    return numeric[0] if len(numeric) == 1 else None


def s_rate_value_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ("S_rate", "S_rates", "S", "rate", "value", "Value", "S_i")
    c = find_col(df, candidates)
    if c is not None:
        return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric:
        if "s" in str(c).lower() or "rate" in str(c).lower():
            return c
    return numeric[0] if len(numeric) == 1 else None


def build_network_genes(parsed: dict[str, Any]) -> list[NetworkGene]:
    gene_df = parsed["gene_df"]
    gene_col = parsed["gene_col"]
    psite_df = parsed.get("psite_df", pd.DataFrame())
    s_df = parsed.get("s_rates_df", pd.DataFrame())
    psite_gene_col = get_gene_column(psite_df) if not psite_df.empty else None
    psite_col = get_psite_column(psite_df) if not psite_df.empty else None
    dp_col = psite_value_column(psite_df) if not psite_df.empty else None
    s_gene_col = get_gene_column(s_df) if not s_df.empty and find_col(s_df, GENE_COL_CANDIDATES) else None
    s_psite_col = get_psite_column(s_df) if not s_df.empty else None
    s_val_col = s_rate_value_column(s_df) if not s_df.empty else None

    genes: list[NetworkGene] = []
    for _, row in gene_df.iterrows():
        name = str(row[gene_col]).strip()
        if not name:
            continue
        g = NetworkGene(
            name=name,
            A=row_value(row, PARAM_ALIASES["A"]),
            B=row_value(row, PARAM_ALIASES["B"]),
            C=row_value(row, PARAM_ALIASES["C"]),
            D=row_value(row, PARAM_ALIASES["D"]),
            E=row_value(row, PARAM_ALIASES["E"]),
            tf_scale=row_value(row, PARAM_ALIASES["tf_scale"], 1.0),
        )
        if psite_gene_col and psite_col:
            sub = psite_df[psite_df[psite_gene_col].astype(str).str.strip() == name]
            for _, prow in sub.iterrows():
                site = str(prow[psite_col]).strip()
                if not site:
                    continue
                g.sites.append(site)
                if dp_col is not None:
                    g.Dp[site] = finite_float(prow[dp_col], row_value(row, ("Phospho_Degradation_Dp_mean",), 0.0))
                else:
                    g.Dp[site] = row_value(row, ("Phospho_Degradation_Dp_mean",), 0.0)
        if not g.sites:
            # Fallback: one aggregate site if only mean Dp is available.
            dp_mean = row_value(row, ("Phospho_Degradation_Dp_mean",), np.nan)
            if not math.isnan(dp_mean):
                g.sites.append("site1")
                g.Dp["site1"] = dp_mean
        if s_gene_col and s_psite_col and s_val_col:
            sub = s_df[s_df[s_gene_col].astype(str).str.strip() == name]
            for _, srow in sub.iterrows():
                site = str(srow[s_psite_col]).strip()
                if site in g.sites:
                    g.S[site] = finite_float(srow[s_val_col], 0.0)
        for site in g.sites:
            g.S.setdefault(site, 0.0)
            g.Dp.setdefault(site, row_value(row, ("Phospho_Degradation_Dp_mean",), 0.0))
        genes.append(g)
    return genes


def infer_network_model_id(results_dir: Path) -> int:
    """
    Infer networkmodel MODEL id from explicit selected-model metadata.

    Mapping:
      0 = distributive
      1 = successive / sequential
      2 = combinatorial
      4 = saturated

    Do not infer from available_models, because that field lists all supported
    models and commonly contains "combinatorial" even for distributive runs.
    """
    def normalize_model_value(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and float(value).is_integer():
            iv = int(value)
            return iv if iv in {0, 1, 2, 4} else None
        text = str(value).strip().lower().strip('"').strip("'")
        if text in {"0", "model0", "model_0", "distributive", "dist", "distmod"}:
            return 0
        if text in {"1", "model1", "model_1", "successive", "sequential", "succ", "succmod"}:
            return 1
        if text in {"2", "model2", "model_2", "combinatorial", "comb", "combinatoric"}:
            return 2
        if text in {"4", "model4", "model_4", "saturated", "saturating", "saturation"}:
            return 4
        return None

    def find_explicit_model_in_json(obj: Any) -> Optional[int]:
        preferred_keys = {
            "model_code", "model_id", "model", "MODEL",
            "network_model_id", "networkmodel_model_id", "state_layout",
        }
        if isinstance(obj, dict):
            for key in preferred_keys:
                if key in obj and not isinstance(obj[key], (list, tuple, dict)):
                    mid = normalize_model_value(obj[key])
                    if mid is not None:
                        return mid
            for key, val in obj.items():
                if str(key).lower() in {"available_models", "supported_models", "model_options"}:
                    continue
                mid = find_explicit_model_in_json(val)
                if mid is not None:
                    return mid
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    mid = find_explicit_model_in_json(item)
                    if mid is not None:
                        return mid
        return None

    for name in ("mode_metadata.json", "metadata.json", "optimized_entities.json"):
        path = results_dir / name
        if path.is_file():
            try:
                mid = find_explicit_model_in_json(json.loads(path.read_text()))
                if mid is not None:
                    return mid
            except Exception:
                pass

    cfg = results_dir / "config_resolved.yaml"
    if cfg.is_file():
        text = cfg.read_text(errors="ignore")
        for pattern in (
            r"(?im)^\s*model_code\s*:\s*([0-9]+)\s*$",
            r"(?im)^\s*model_id\s*:\s*([0-9]+)\s*$",
            r"(?im)^\s*MODEL\s*:\s*([0-9]+)\s*$",
        ):
            m = re.search(pattern, text)
            if m:
                mid = normalize_model_value(m.group(1))
                if mid is not None:
                    return mid
        for pattern in (
            r"(?im)^\s*model\s*:\s*([A-Za-z0-9_\-]+)\s*$",
            r"(?im)^\s*state_layout\s*:\s*([A-Za-z0-9_\-]+)\s*$",
        ):
            m = re.search(pattern, text)
            if m:
                mid = normalize_model_value(m.group(1))
                if mid is not None:
                    return mid

    low = str(results_dir).lower()
    if "combinatorial" in low:
        return 2
    if "successive" in low or "sequential" in low:
        return 1
    if "saturated" in low or "saturating" in low:
        return 4
    if "distributive" in low:
        return 0
    return 0


def initial_conditions(results_dir: Path, species_order: list[str]) -> dict[str, float]:
    p = results_dir / "initial_conditions_y0.csv"
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        try:
            df = pd.read_csv(p, header=None)
        except Exception:
            return {}
    # Long schema with id/value.
    cols = {str(c).lower(): c for c in df.columns}
    id_col = None
    for c in ("species", "state", "id", "name", "sbml_id"):
        if c in cols:
            id_col = cols[c]
            break
    val_col = None
    for c in ("value", "y0", "initial", "initial_condition"):
        if c in cols:
            val_col = cols[c]
            break
    if id_col is not None and val_col is not None:
        return {safe_id(r[id_col], "s"): finite_float(r[val_col]) for _, r in df.iterrows()}
    # Vector schema.
    arr = df.select_dtypes(include=[np.number]).to_numpy().reshape(-1)
    if arr.size == len(species_order):
        return {sid: finite_float(arr[i]) for i, sid in enumerate(species_order)}
    return {}


# ---------------------------------------------------------------------------
# SBML builders
# ---------------------------------------------------------------------------

def make_document(model_id: str, model_name: str):
    ensure_libsbml()
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(safe_id(model_id, "model"))
    model.setName(model_name)
    try:
        model.setSubstanceUnits("dimensionless")
        model.setTimeUnits("dimensionless")
        model.setExtentUnits("dimensionless")
    except Exception:
        pass
    cell_compartment(model)
    return doc, model


def export_kin_tf_to_sbml(results_dir: Path, output_dir: Path, family: str, validate: bool) -> list[dict[str, Any]]:
    workbook_name = "kinopt_results.xlsx" if family == "kinopt" else "tfopt_results.xlsx"
    wb = first_existing(results_dir, (workbook_name,))
    if wb is None:
        raise FileNotFoundError(f"Could not find {workbook_name} in {results_dir} or {results_dir/'tables'}")
    alpha, beta = parse_kinopt_xlsx(wb) if family == "kinopt" else parse_tfopt_xlsx(wb)
    doc, model = make_document(family, family)
    add_notes(model, [
        f"Generated from {wb}.",
        "Only fitted coefficient sheets were parsed: Alpha Values and Beta Values.",
        "All values are dimensionless fold-change/regulatory coefficients.",
    ])
    if family == "kinopt":
        for _, row in alpha.iterrows():
            pid = safe_id(f"alpha_{row['Gene']}_{row['Psite']}_{row['Kinase']}", "alpha")
            create_parameter(model, pid, row["Alpha"], name=f"alpha {row['Gene']} {row['Psite']} {row['Kinase']}")
        for _, row in beta.iterrows():
            pid = safe_id(f"beta_{row['Kinase']}_{row['Psite']}", "beta")
            create_parameter(model, pid, row["Beta"], name=f"beta {row['Kinase']} {row['Psite']}")
    else:
        for _, row in alpha.iterrows():
            pid = safe_id(f"alpha_{row['mRNA']}_{row['TF']}", "alpha")
            create_parameter(model, pid, row["Value"], name=f"alpha {row['mRNA']} {row['TF']}")
        for _, row in beta.iterrows():
            pid = safe_id(f"beta_{row['TF']}_{row['PSite']}", "beta")
            create_parameter(model, pid, row["Value"], name=f"beta {row['TF']} {row['PSite']}")
    return [write_doc(doc, output_dir / f"{family}.xml", validate)]


def export_protwise_to_sbml(results_dir: Path, output_dir: Path, validate: bool) -> list[dict[str, Any]]:
    parsed = parse_protwise_parameters(results_dir)
    if not parsed["gene_params"]:
        raise ValueError(f"No <GENE>_parameters.xlsx files found in {results_dir}")
    summaries = []
    for entry in parsed["gene_params"]:
        gene = entry["gene"]
        params = entry["params"]
        model_kind = detect_protwise_model(params)
        doc, model = make_document(gene, f"Protwise {gene}")
        add_notes(model, [
            f"Generated from {entry['path']}.",
            "Equations grounded in protwise/models/diffrax_solver.py and legacy protwise.models.* modules.",
            "All state variables and fitted rates are dimensionless; time is treated as dimensionless experiment time.",
        ])
        for pname, val in params.items():
            create_parameter(model, safe_id(pname, "p"), val, name=pname)
        s_names = sorted([n for n in params if re.fullmatch(r"S\d+", str(n))], key=lambda x: int(str(x)[1:]))
        d_names = sorted([n for n in params if re.fullmatch(r"D\d+", str(n))], key=lambda x: int(str(x)[1:]))
        n = len(s_names)
        species_order = []
        for sid in ("R", "P"):
            create_species(model, sid, sid, 0.0)
            species_order.append(sid)
        if model_kind == "randmod":
            for idx in range(1, n + 1):
                sid = f"X{idx}"
                create_species(model, sid, sid, 0.0)
            # Do not expand combinatorial randmod unless source state schema is explicit.
        else:
            for idx in range(1, n + 1):
                sid = f"X{idx}"
                create_species(model, sid, sid, 0.0)
            add_rate_rule(model, "R", "A - B*R")
            sum_s = " + ".join(s_names) if s_names else "0"
            sum_x = " + ".join(f"X{i}" for i in range(1, n + 1)) if n else "0"
            add_rate_rule(model, "P", f"C*R - (D + {sum_s})*P + {sum_x}")
            for i in range(1, n + 1):
                s = f"S{i}"
                d = f"D{i}"
                x = f"X{i}"
                add_rate_rule(model, x, f"{s}*P - (1 + {d})*{x}")
        summaries.append(write_doc(doc, output_dir / f"{safe_id(gene)}.xml", validate))
    return summaries


def export_networkmodel_to_sbml(results_dir: Path, output_dir: Path, validate: bool) -> list[dict[str, Any]]:
    parsed = parse_networkmodel_tables(results_dir)
    genes = build_network_genes(parsed)
    if not genes:
        raise ValueError("No networkmodel genes could be parsed from model_parameters_genes.csv")
    model_id = infer_network_model_id(results_dir)
    doc, model = make_document("networkmodel", "Networkmodel")
    add_notes(model, [
        "Generated from networkmodel fitted result directory.",
        "Equations grounded in networkmodel/backend.py::make_networkmodel_rhs.",
        f"Inferred backend MODEL={model_id}. MODEL=0 uses distributive phosphorylation, MODEL=1 uses sequential/successive phosphorylation, MODEL=2 uses combinatorial phosphorylation states, and MODEL=4 uses saturated translation/phosphorylation.",
        "S_ij values are read from S_rates_picked.csv when available; otherwise they are exported as zero-valued constants with warnings in console output.",
        "All states, fitted parameters, fold changes and time are represented as dimensionless.",
    ])

    species_order: list[str] = []
    # First pass: parameters and species.
    for g in genes:
        gs = safe_id(g.name, "g")
        create_parameter(model, f"A_{gs}", g.A, name=f"{g.name} Synthesis_A")
        create_parameter(model, f"B_{gs}", g.B, name=f"{g.name} mRNA_Degradation_B")
        create_parameter(model, f"C_{gs}", g.C, name=f"{g.name} Translation_C")
        create_parameter(model, f"D_{gs}", g.D, name=f"{g.name} Protein_Degradation_D")
        create_parameter(model, f"E_{gs}", g.E, name=f"{g.name} De-Phosphorylation_E")
        create_parameter(model, f"tf_scale_{gs}", g.tf_scale, name=f"{g.name} Global_TF_Scale")
        sid_R = f"R_{gs}"
        create_species(model, sid_R, f"{g.name} mRNA", 0.0)
        species_order.append(sid_R)
        if model_id == 2:
            n = len(g.sites)
            max_states = 2 ** n if n <= 12 else 0
            if max_states == 0:
                # Safety fallback for very large combinatorial systems.
                sid_P = f"P_{gs}"
                create_species(model, sid_P, f"{g.name} total protein", 0.0)
                species_order.append(sid_P)
            else:
                for m in range(max_states):
                    sid = f"P_{gs}_m{m}"
                    create_species(model, sid, f"{g.name} phospho-mask {m}", 0.0)
                    species_order.append(sid)
        else:
            sid_P = f"P_{gs}"
            create_species(model, sid_P, f"{g.name} protein", 0.0)
            species_order.append(sid_P)
            for site in g.sites:
                ss = safe_id(site, "site")
                sid = f"X_{gs}_{ss}"
                create_species(model, sid, f"{g.name} {site}", 0.0)
                species_order.append(sid)
        for site in g.sites:
            ss = safe_id(site, "site")
            create_parameter(model, f"Dp_{gs}_{ss}", g.Dp.get(site, 0.0), name=f"{g.name} {site} Dp_i")
            create_parameter(model, f"S_{gs}_{ss}", g.S.get(site, 0.0), name=f"{g.name} {site} S_rate")

    # Apply initial conditions if order matches initial_conditions_y0.csv.
    y0 = initial_conditions(results_dir, species_order)
    for sid, val in y0.items():
        sp = model.getSpecies(sid)
        if sp is not None:
            sp.setInitialAmount(finite_float(val))

    # Assignment rules for total protein and simplified TF synthesis.
    for g in genes:
        gs = safe_id(g.name, "g")
        if model_id == 2 and len(g.sites) <= 12:
            total_terms = [f"P_{gs}_m{m}" for m in range(2 ** len(g.sites))]
        else:
            total_terms = [f"P_{gs}"] + [f"X_{gs}_{safe_id(site, 'site')}" for site in g.sites if model_id != 2]
        ptotal = f"Ptotal_{gs}"
        create_parameter(model, ptotal, 0.0, name=f"{g.name} total protein", constant=False)
        add_assignment_rule(model, ptotal, " + ".join(total_terms) if total_terms else "0")
        # Use a self-driven TF input if no reliable TF matrix mapping is available.
        tf_input = f"TF_input_{gs}"
        u = f"u_{gs}"
        synth = f"synth_{gs}"
        create_parameter(model, tf_input, 0.0, name=f"{g.name} TF input", constant=False)
        create_parameter(model, u, 0.0, name=f"{g.name} saturated TF input", constant=False)
        create_parameter(model, synth, 0.0, name=f"{g.name} synthesis term", constant=False)
        # Placeholder with self total protein preserves a connected regulatory term in SBML;
        # source network matrices remain documented in notes.
        add_assignment_rule(model, tf_input, ptotal)
        add_assignment_rule(model, u, f"{tf_input}/(1 + abs({tf_input}))")
        add_assignment_rule(
            model,
            synth,
            f"piecewise(A_{gs}*(1 + tf_scale_{gs}*{u}/(1 + {u} + 1e-6)), geq({u}, 0), A_{gs}/(1 + tf_scale_{gs}*abs({u})))",
        )

    # Rate rules from backend.py.
    for g in genes:
        gs = safe_id(g.name, "g")
        add_rate_rule(model, f"R_{gs}", f"synth_{gs} - B_{gs}*R_{gs}")
        if model_id == 2 and len(g.sites) <= 12:
            n = len(g.sites)
            for m in range(2 ** n):
                sid = f"P_{gs}_m{m}"
                terms: list[str] = []
                if m == 0:
                    terms.append(f"C_{gs}*R_{gs} - D_{gs}*{sid}")
                for j, site in enumerate(g.sites):
                    ss = safe_id(site, "site")
                    bit = 1 << j
                    if m & bit:
                        src = f"P_{gs}_m{m ^ bit}"
                        terms.append(f"S_{gs}_{ss}*{src}")
                        terms.append(f"- E_{gs}*{sid}")
                        terms.append(f"- (Dp_{gs}_{ss} + D_{gs})*{sid}")
                    else:
                        src = f"P_{gs}_m{m | bit}"
                        terms.append(f"E_{gs}*{src}")
                        terms.append(f"- S_{gs}_{ss}*{sid}")
                add_rate_rule(model, sid, " + ".join(terms) if terms else "0")
        else:
            P = f"P_{gs}"
            if model_id == 4:
                trans = f"C_{gs}*R_{gs}/(1 + R_{gs})"
                fwd = lambda ss: f"S_{gs}_{ss}*{P}/(1 + {P})"
            else:
                trans = f"C_{gs}*R_{gs}"
                fwd = lambda ss: f"S_{gs}_{ss}*{P}"
            fwd_terms = []
            back_terms = []
            for site in g.sites:
                ss = safe_id(site, "site")
                X = f"X_{gs}_{ss}"
                f = fwd(ss)
                fwd_terms.append(f)
                back_terms.append(f"E_{gs}*{X}")
                add_rate_rule(model, X, f"{f} - (E_{gs} + Dp_{gs}_{ss} + D_{gs})*{X}")
            add_rate_rule(
                model,
                P,
                f"{trans} - D_{gs}*{P} - ({' + '.join(fwd_terms) if fwd_terms else '0'}) + ({' + '.join(back_terms) if back_terms else '0'})",
            )

    summary = write_doc(doc, output_dir / "networkmodel.xml", validate)
    summary["parsed_files"] = [str(p) for p in parsed["files_parsed"]]
    summary["skipped_files"] = [str(p) for p in parsed["files_skipped"]]
    summary["model_id"] = model_id
    summary["n_genes"] = len(genes)
    summary["n_species"] = len(species_order)
    return [summary]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Export fitted PhosKinTime results to SBML Level 3 Version 2.")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-family", choices=["networkmodel", "protwise", "kinopt", "tfopt"], default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        # Non-destructive default; individual file overwrites are avoided unless requested.
        pass
    output_dir.mkdir(parents=True, exist_ok=True)

    family, family_source = resolve_family(results_dir, args.model_family)
    if args.verbose:
        print(f"Model family: {family} ({family_source})")
        print(f"Results dir: {results_dir}")
        print(f"Output dir: {output_dir}")

    if family in {"kinopt", "tfopt"}:
        summaries = export_kin_tf_to_sbml(results_dir, output_dir, family, args.validate)
    elif family == "protwise":
        summaries = export_protwise_to_sbml(results_dir, output_dir, args.validate)
    elif family == "networkmodel":
        summaries = export_networkmodel_to_sbml(results_dir, output_dir, args.validate)
    else:
        raise ValueError(f"Unsupported family: {family}")

    print("\nFinal summary")
    print("Implementation grounding:")
    if family == "networkmodel":
        print("- networkmodel equations grounded in networkmodel/backend.py::make_networkmodel_rhs")
    elif family == "protwise":
        print("- protwise equations grounded in protwise/models/diffrax_solver.py and legacy protwise.models.* modules")
    else:
        print("- KinOpt/TFOpt schemas grounded in exporter sheet functions")
    print("\nResult parsing:")
    for item in summaries:
        for p in item.get("parsed_files", []):
            print(f"- parsed: {p}")
        for p in item.get("skipped_files", []):
            print(f"- skipped: {p}")
    print("\nOutputs:")
    for item in summaries:
        print(f"- SBML: {item['path']}")
        if args.validate:
            print(f"  validation: {item['errors']} errors, {item['warnings']} warnings")
            if args.verbose:
                for msg in item["messages"][:10]:
                    print(f"  - {msg}")
    print("- SBML files written and validated where possible")


if __name__ == "__main__":
    main()