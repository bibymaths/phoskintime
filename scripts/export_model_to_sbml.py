#!/usr/bin/env python3
"""Export fitted PhosKinTime-family model results to dimensionless SBML."""
from __future__ import annotations

import argparse, csv, json, logging, math, pickle, re, sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

FAMILIES = ("protwise", "kinopt", "tfopt", "networkmodel")
PARAM_HINTS = ("param", "parameter", "alpha", "beta", "theta", "coef", "weight", "rate", "results")
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".pkl", ".pickle", ".npz", ".npy", ".parquet", ".xlsx", ".xls"}
SOURCE_HINTS = {
    "protwise": ["protwise/models/protwise.py", "protwise/models/diffrax_solver.py", "protwise/paramest/core.py", "config/constants.py"],
    "kinopt": ["kinopt/local/objfn/minfn.py", "kinopt/local/utils/params.py", "kinopt/local/optcon/construct.py", "kinopt/local/exporter/sheetutils.py"],
    "tfopt": ["tfopt/local/objfn/minfn.py", "tfopt/local/utils/params.py", "tfopt/local/optcon/construct.py"],
    "networkmodel": ["networkmodel/models.py", "networkmodel/params.py", "networkmodel/export.py", "networkmodel/LossFunction.py"],
}
EQUATIONS = {
    "kinopt": ["M_k(t) = sum_s beta_{k,s} K_{k,s}(t)", "P_i(t) = max(0, sum_k alpha_{i,k} M_k(t))"],
    "tfopt": ["E_g(t) = max(0, sum_r alpha_{g,r} (beta_{r,0} TF_r(t) + sum_s beta_{r,s} PSite_{r,s}(t)))"],
    "protwise": ["dR_i/dt = transcription_i(t) - degradation_i R_i", "dP_i/dt = translation_i(R_i) - degradation_i P_i - phosphorylation flux + dephosphorylation flux"],
    "networkmodel": ["dR_i/dt = synthesis_i(TF inputs) - B_i R_i", "dP_i/dt = translation_i(R_i) - D_i P_i - site fluxes", "dS_{i,j}/dt = phosphorylation_{i,j}(P_i) - (D_i + Dp_{i,j})S_{i,j} - dephosphorylation_{i,j}"],
}

@dataclass
class ModelResult:
    model_family: str
    source_path: Path
    parameters: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    equations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

@dataclass
class ValidationReport:
    path: Path
    errors: int
    warnings: int
    messages: list[str]


def infer_model_family(path: Path) -> str | None:
    low = str(path).lower()
    for fam in FAMILIES:
        if fam in low:
            return fam
    return None


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _flatten(prefix: str, obj: Any, out: dict[str, float], limit: int = 5000) -> None:
    if len(out) >= limit: return
    if isinstance(obj, dict):
        for k, v in obj.items(): _flatten(f"{prefix}_{k}" if prefix else str(k), v, out, limit)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): _flatten(f"{prefix}_{i}", v, out, limit)
    else:
        f = _as_float(obj)
        if f is not None and prefix: out[_safe_id(prefix)] = f


def _safe_id(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(text).strip())[:180]
    if not s or s[0].isdigit(): s = "p_" + s
    return s


def _read_table(path: Path) -> dict[str, float]:
    params: dict[str, float] = {}
    try:
        if path.suffix.lower() in {".csv", ".tsv"}:
            dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                rows = list(csv.DictReader(fh, dialect=dialect))
        else:
            try:
                import pandas as pd
                frames = pd.read_excel(path, sheet_name=None) if path.suffix.lower() in {".xlsx", ".xls"} else {"table": pd.read_parquet(path)}
                rows = []
                for sheet, df in frames.items():
                    for rec in df.to_dict("records"):
                        rec = {f"{sheet}_{k}": v for k, v in rec.items()} | rec
                        rows.append(rec)
            except Exception as exc:
                return {"__warning__": f"Could not read table {path}: {exc}"}  # type: ignore[dict-item]
        for n, row in enumerate(rows):
            keys = list(row)
            val_cols = [k for k in keys if _as_float(row.get(k)) is not None]
            name_bits = [str(row.get(k)) for k in keys if k not in val_cols and row.get(k) not in (None, "")][:4]
            for col in val_cols:
                if any(h in col.lower() for h in PARAM_HINTS) or any(h in path.stem.lower() for h in PARAM_HINTS):
                    params[_safe_id("_".join(name_bits + [col]) or f"row{n}_{col}")] = float(row[col])
    except Exception as exc:
        params["__warning__"] = f"Could not read {path}: {exc}"  # type: ignore[assignment]
    return params


def load_model_result(path: Path) -> ModelResult:
    fam = infer_model_family(path) or "unknown"
    params: dict[str, float] = {}; warnings: list[str] = []
    try:
        suf = path.suffix.lower()
        if suf == ".json": _flatten("", json.loads(path.read_text(encoding="utf-8")), params)
        elif suf in {".pkl", ".pickle"}: _flatten("", pickle.loads(path.read_bytes()), params)
        elif suf in {".npz", ".npy"}:
            import numpy as np
            obj = np.load(path, allow_pickle=True)
            if hasattr(obj, "files"):
                for k in obj.files: _flatten(k, obj[k].tolist(), params)
            else: _flatten(path.stem, obj.tolist(), params)
        elif suf in {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}: params.update(_read_table(path))
    except Exception as exc:
        warnings.append(f"Could not load {path}: {exc}")
    warn = params.pop("__warning__", None)
    if warn: warnings.append(str(warn))
    if not params: warnings.append("No numeric fitted parameters detected in this file.")
    return ModelResult(fam, path, params, {"result_file": str(path)}, EQUATIONS.get(fam, []), warnings)


def discover_model_outputs(results_dir: Path, model_family: str | None = None, run_id: str | None = None) -> list[ModelResult]:
    if not results_dir.exists(): raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    root = results_dir / run_id if run_id else results_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    out = []
    for p in files:
        fam = infer_model_family(p)
        if model_family and (fam or "unknown").lower() != model_family.lower(): continue
        if fam or any(h in p.name.lower() for h in PARAM_HINTS): out.append(load_model_result(p))
    # merge per family
    merged: dict[str, ModelResult] = {}
    for r in out:
        if r.model_family == "unknown": continue
        m = merged.setdefault(r.model_family, ModelResult(r.model_family, r.source_path, {}, {"result_files": []}, EQUATIONS.get(r.model_family, []), []))
        m.metadata["result_files"].append(str(r.source_path)); m.parameters.update(r.parameters); m.warnings.extend(r.warnings)
    return list(merged.values())


def build_sbml_document(model_result: ModelResult):
    try:
        import libsbml  # type: ignore
    except Exception:
        return None
    doc = libsbml.SBMLDocument(3, 2); model = doc.createModel(); model.setId(_safe_id(model_result.model_family + "_model")); model.setName(model_result.model_family)
    model.setTimeUnits("dimensionless"); model.setExtentUnits("dimensionless"); model.setSubstanceUnits("dimensionless")
    comp = model.createCompartment(); comp.setId("unitless_compartment"); comp.setConstant(True); comp.setSize(1.0); comp.setSpatialDimensions(0); comp.setUnits("dimensionless")
    notes = """<body xmlns='http://www.w3.org/1999/xhtml'><p>This SBML model was exported from fitted phoskintime-family results. The fitted observables are fold-change or otherwise unitless measurements. Species and parameters corresponding to these observables are represented as dimensionless quantities. The model is a mathematical regulatory/phosphorylation model, not an absolute concentration-based biochemical model unless additional calibration information is supplied. Time is represented using a dimensionless convention because source result units may be unknown.</p></body>"""
    model.setNotes(notes)
    obs = model.createSpecies(); obs.setId("unitless_observable"); obs.setCompartment("unitless_compartment"); obs.setInitialAmount(0.0); obs.setSubstanceUnits("dimensionless"); obs.setBoundaryCondition(False); obs.setHasOnlySubstanceUnits(False); obs.setConstant(False)
    for name, value in sorted(model_result.parameters.items()):
        p = model.createParameter(); p.setId(_safe_id(name)); p.setValue(float(value)); p.setUnits("dimensionless"); p.setConstant(True)
    for i, eq in enumerate(model_result.equations):
        ar = model.createAssignmentRule(); ar.setVariable("unitless_observable"); ar.setMath(libsbml.parseL3Formula("unitless_observable" if i else "0")); ar.setNotes(f"<body xmlns='http://www.w3.org/1999/xhtml'><p>Source equation: {escape(eq)}</p></body>")
        break
    return doc


def _write_fallback_sbml(model_result: ModelResult, output_path: Path) -> None:
    ns = "http://www.sbml.org/sbml/level3/version2/core"; ET.register_namespace("", ns)
    sbml = ET.Element(f"{{{ns}}}sbml", {"level":"3", "version":"2"}); model = ET.SubElement(sbml, f"{{{ns}}}model", {"id":_safe_id(model_result.model_family+"_model"), "substanceUnits":"dimensionless", "timeUnits":"dimensionless", "extentUnits":"dimensionless"})
    notes = ET.SubElement(model, f"{{{ns}}}notes"); body = ET.SubElement(notes, "body", {"xmlns":"http://www.w3.org/1999/xhtml"}); ET.SubElement(body,"p").text = "Fold-change/unitless fitted observables are represented with dimensionless units; this is a mathematical regulatory/phosphorylation model, not an absolute concentration model."
    lop = ET.SubElement(model, f"{{{ns}}}listOfParameters")
    for k,v in sorted(model_result.parameters.items()): ET.SubElement(lop, f"{{{ns}}}parameter", {"id":_safe_id(k), "value":repr(float(v)), "units":"dimensionless", "constant":"true"})
    ET.ElementTree(sbml).write(output_path, encoding="utf-8", xml_declaration=True)


def write_sbml(document, model_result: ModelResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if document is None: _write_fallback_sbml(model_result, output_path)
    else:
        import libsbml  # type: ignore
        libsbml.writeSBMLToFile(document, str(output_path))


def validate_sbml(output_path: Path) -> ValidationReport:
    try:
        import libsbml  # type: ignore
    except Exception:
        return ValidationReport(output_path, 0, 1, ["python-libsbml is not installed; install it with `pip install python-libsbml` to perform SBML consistency validation."])
    doc = libsbml.readSBML(str(output_path)); doc.checkConsistency()
    errs = warns = 0; msgs=[]
    for i in range(doc.getNumErrors()):
        e = doc.getError(i); sev = e.getSeverity(); msgs.append(f"{e.getSeverityAsString()}: {e.getMessage().strip()}")
        if sev >= libsbml.LIBSBML_SEV_ERROR: errs += 1
        else: warns += 1
    return ValidationReport(output_path, errs, warns, msgs)


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--results-dir", required=True, type=Path); ap.add_argument("--output-dir", required=True, type=Path); ap.add_argument("--model-family", choices=FAMILIES); ap.add_argument("--run-id"); ap.add_argument("--validate", action="store_true"); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--verbose", action="store_true")
    a=ap.parse_args(argv); logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    results=discover_model_outputs(a.results_dir, a.model_family, a.run_id); reports=[]; exported=[]
    found={r.model_family for r in results}
    for fam in FAMILIES:
        if a.model_family and fam != a.model_family: continue
        if fam not in found: logging.warning("%s: skipped, no fitted result files found", fam)
    for r in results:
        if not r.parameters: logging.warning("%s: skipped, no parameters found in %s", r.model_family, r.source_path); continue
        out=a.output_dir / f"{r.model_family}_model.xml"
        if out.exists() and not a.overwrite: logging.warning("%s exists; use --overwrite to replace", out); continue
        write_sbml(build_sbml_document(r), r, out); exported.append(out)
        rep = validate_sbml(out) if a.validate else ValidationReport(out,0,0,["Validation not requested; pass --validate."]); reports.append(rep)
        print(f"{out}: {rep.errors} errors, {rep.warnings} warnings"); [print(f"  - {m}") for m in rep.messages[:20]]
    print("Discovered model families:"); [print(f"- {fam}: {'exported SBML' if any(p.name.startswith(fam) for p in exported) else 'skipped'}") for fam in FAMILIES if not a.model_family or fam==a.model_family]
    print("Outputs:"); [print(f"- {p}") for p in exported]
    return 0
if __name__ == "__main__": raise SystemExit(main())
