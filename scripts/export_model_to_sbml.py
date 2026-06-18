#!/usr/bin/env python3
"""Export fitted PhosKinTime-family model results to dimensionless SBML."""
from __future__ import annotations

import argparse, csv, json, logging, math, pickle, re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

# Schema source of truth inspected before implementing the parser:
# - kinopt/local/exporter/sheetutils.py::output_results writes workbook sheets
#   "Alpha Values" with columns Gene, Psite, Kinase, Alpha and "Beta Values"
#   with columns Kinase, Psite, Beta.
# - tfopt/local/exporter/sheetutils.py::output_results writes workbook sheets
#   "Alpha Values" and "Beta Values"; app/tfopt.py::load_alpha/load_beta define
#   accepted normalized columns mRNA, TF, Value and TF, PSite, Value.
# - common/utils/display.py::save_result writes ProtWise sheets named
#   "<gene>_params" with Gene, Time(min), fitted parameter columns,
#   Regularization, and optional MSE; non-parameter sheets include *_errors,
#   *_solution, *_site_estimates, *_site_observed, *_pca, *_tsne.
# - protwise/paramest/core.py::process_gene writes per-gene
#   "<gene>_parameters.xlsx" with Time, fitted parameter columns, and
#   Regularization.
# - networkmodel/export.py::export_pareto_front_to_excel writes sheets
#   params_genes(sol_id, protein, param, value), params_kinases(sol_id, kinase,
#   c_k), deg_sites(sol_id, site_id, protein, psite, local_site, k_deg), and
#   non-parameter summary/traj_* sheets.
# - networkmodel/dashboard_bundle.py::save_dashboard_bundle stores raw optimizer
#   pareto_X/parameter_values in artifacts/dashboard_bundle.pkl, but table exports
#   above are preferred because they preserve named physical parameters.
# - dashboard/result_parser.py::discover_result_directory treats tables/ and
#   legacy kinopt_results.xlsx/tfopt_results.xlsx as dashboard-readable tables.

FAMILIES = ("protwise", "kinopt", "tfopt", "networkmodel")
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".pkl", ".pickle", ".npz", ".npy", ".parquet", ".xlsx", ".xls", ".tex"}
SOURCE_HINTS = {
    "protwise": ["protwise/models/protwise.py", "protwise/models/diffrax_solver.py", "protwise/paramest/core.py", "common/utils/display.py", "config/constants.py"],
    "kinopt": ["kinopt/local/objfn/minfn.py", "kinopt/local/utils/params.py", "kinopt/local/optcon/construct.py", "kinopt/local/exporter/sheetutils.py", "app/kinopt.py"],
    "tfopt": ["tfopt/local/objfn/minfn.py", "tfopt/local/utils/params.py", "tfopt/local/optcon/construct.py", "tfopt/local/exporter/sheetutils.py", "app/tfopt.py"],
    "networkmodel": ["networkmodel/models.py", "networkmodel/params.py", "networkmodel/export.py", "networkmodel/dashboard_bundle.py", "networkmodel/dashboard_app.py"],
}
SCHEMA_SOURCES = [
    "kinopt/local/exporter/sheetutils.py::output_results",
    "tfopt/local/exporter/sheetutils.py::output_results",
    "app/kinopt.py::parse_edges/parse_betas",
    "app/tfopt.py::load_alpha/load_beta",
    "common/utils/display.py::save_result",
    "protwise/paramest/core.py::process_gene",
    "networkmodel/export.py::export_pareto_front_to_excel",
    "networkmodel/dashboard_bundle.py::save_dashboard_bundle",
    "dashboard/result_parser.py::discover_result_directory",
]
EQUATION_SOURCES = {
    "kinopt": ["kinopt/local/objfn/minfn.py::_estimated_series", "kinopt/local/objfn/minfn.py::_objective"],
    "tfopt": ["tfopt/local/objfn/minfn.py::compute_predictions", "tfopt/local/objfn/minfn.py::objective_"],
    "protwise": ["protwise/models/diffrax_solver.py::_dist_rhs/_succ_rhs/_rand_rhs", "protwise/models/diffrax_solver.py::make_local_model_rhs", "config/constants.py::get_param_names"],
    "networkmodel": ["networkmodel/models.py::calculate_synthesis_rate", "networkmodel/models.py::saturating_rhs/distributive_rhs/sequential_rhs/combinatorial_rhs", "networkmodel/LossFunction.py::loss_function_noncomb", "networkmodel/params.py::unpack_params"],
    "unknown": [],
}
EQUATIONS = {
    "kinopt": ["A_k(t) = \\sum_s beta_{k,s} K_{k,s}(t)", "P_i(t) = max(0, \\sum_k alpha_{i,k} A_k(t))", "Loss is computed from observed minus predicted phosphorylation/protein fold-change with configured robust loss option."],
    "tfopt": ["TFEffect_r(t) = beta_{r,0} TF_r(t) + \\sum_s beta_{r,s} PSite_{r,s}(t)", "E_g(t) = max(0, \\sum_r alpha_{g,r} TFEffect_r(t))", "Loss is mean residual loss over observed gene-expression fold-change plus optional regularization."],
    "protwise": ["dist/protwise: dR/dt = A - B R; dP/dt = C R - (D + \\sum_j s_j)P + \\sum_j X_j; dX_j/dt = s_j P - (1+d_j)X_j", "succmod: sequential phosphosite states use upstream phosphorylation and unit dephosphorylation terms as implemented in _succ_rhs.", "randmod: subset-state phosphorylation/dephosphorylation ODEs are implemented in _rand_rhs and aggregated to site-level output."],
    "networkmodel": ["synthesis_i = calculate_synthesis_rate(A_i, tf_scale, TF_inputs_i)", "dR_i/dt = synthesis_i - B_i R_i", "non-combinatorial dP_i/dt and dS_{i,j}/dt follow saturating/distributive/sequential_rhs; combinatorial_rhs uses mask-state phosphorylation transitions.", "Fitted physical parameters are c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, and tf_scale after softplus unpacking."],
    "unknown": ["Project-style fitted parameter table was detected, but model family was not inferred; pass --model-family for family-specific equations."],
}
PARAMETER_SOURCE_PATTERNS = ["parameter", "parameters", "param", "params", "alpha", "beta", "coefficient", "coefficients", "weight", "weights", "fit_params", "fitted_params", "estimated_params"]
NON_PARAMETER_SOURCE_PATTERNS = ["trajectory", "trajectories", "observed", "estimated", "prediction", "predicted", "residual", "residuals", "metric", "metrics", "loss", "objective", "score", "scores", "summary", "fit_summary", "diagnostic", "diagnostics", "plot", "plots", "timecourse", "time_course", "timeseries", "time_series", "fc", "foldchange", "fold_change", "solution", "pca", "tsne", "errors", "traj"]
PARAMETER_NAME_COLUMNS = ["parameter", "param", "name", "parameter_name", "term"]
VALUE_COLUMNS = ["value", "estimate", "estimated_value", "parameter_value", "fitted_value", "coef", "coefficient", "weight"]
ALPHA_VALUE_COLUMNS = ["alpha", "alpha_value", "alpha_estimate"]
BETA_VALUE_COLUMNS = ["beta", "beta_value", "beta_estimate"]
ALPHA_ID_COLUMNS = ["protein", "target", "substrate", "gene", "geneid", "mrna", "kinase", "tf", "regulator", "site", "psite", "phosphosite"]
BETA_ID_COLUMNS = ["kinase", "tf", "regulator", "site", "psite", "phosphosite", "protein", "gene", "geneid"]
WIDE_PARAMETER_PREFIXES = ["alpha_", "beta_", "theta_", "gamma_", "param_", "coef_", "weight_", "k_"]
NEVER_PARAMETER_COLUMNS = {"time", "time_min", "time(min)", "replicate", "observed", "estimated", "predicted", "prediction", "residual", "rmse", "mae", "r2", "score", "id", "index", "mse", "regularization", "sol_id", "rank", "scalar_score", "w_prot", "w_rna", "w_phos", "site_id", "local_site"}


class _MiniBoolSeries:
    def __init__(self, values): self.values = list(values)
    def any(self): return any(self.values)

class _MiniSeries:
    def __init__(self, values): self.values = list(values)
    def map(self, func): return _MiniSeries([func(v) for v in self.values])
    def notna(self): return _MiniBoolSeries([v is not None for v in self.values])

class _MiniILoc:
    def __init__(self, frame): self.frame = frame
    def __getitem__(self, idx): return self.frame.rows[idx]

class _MiniFrame:
    def __init__(self, rows):
        self.rows = list(rows)
        self.columns = list(self.rows[0].keys()) if self.rows else []
        self.iloc = _MiniILoc(self)
    def iterrows(self):
        for i, row in enumerate(self.rows):
            yield i, row
    def __getitem__(self, col):
        return _MiniSeries([row.get(col) for row in self.rows])

@dataclass
class ParameterRecord:
    parameter_id: str
    value: float
    source_file: str
    source_sheet: str | None
    schema: str
    original: dict[str, Any] = field(default_factory=dict)

@dataclass
class SkippedTable:
    file: str
    sheet: str | None
    reason: str

@dataclass
class ModelResult:
    model_family: str
    source_path: Path
    parameters: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    equations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameter_records: list[ParameterRecord] = field(default_factory=list)
    skipped_tables: list[SkippedTable] = field(default_factory=list)
    equation_source_files: list[Path] = field(default_factory=list)
    equation_source_functions: list[str] = field(default_factory=list)
    parameter_source_files: list[Path] = field(default_factory=list)
    schema_source_functions: list[str] = field(default_factory=list)
    structure_files: list[Path] = field(default_factory=list)

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


def assign_model_family(path: Path, requested: str | None) -> tuple[str | None, str, str | None]:
    inferred = infer_model_family(path)
    if requested is not None:
        if inferred is not None and inferred.lower() != requested.lower():
            return None, "conflict", inferred
        return requested, "--model-family", inferred
    if inferred is not None:
        return inferred, "path", inferred
    return "unknown", "unknown", None


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _safe_id(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(text).strip())[:220]
    if not s or s[0].isdigit():
        s = "p_" + s
    return s


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def source_is_non_parameter(source_name: str) -> bool:
    n = _norm(source_name)
    if any(p in n for p in ("fitted_params", "estimated_params", "model_parameters", "parameters", "alpha", "beta")):
        return False
    return any(p in n for p in NON_PARAMETER_SOURCE_PATTERNS)


def source_is_parameter_like(source_name: str) -> bool:
    n = _norm(source_name)
    if n == "results" or n.endswith("_results"):
        return False
    return any(p in n for p in PARAMETER_SOURCE_PATTERNS)


def _read_dataframe_file(path: Path) -> list[tuple[str | None, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            return [(None, _MiniFrame(csv.DictReader(fh, dialect=dialect)))]
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd
        frames = pd.read_excel(path, sheet_name=None)
        return list(frames.items())
    if suffix == ".parquet":
        import pandas as pd
        return [(None, pd.read_parquet(path))]
    return []


def _row_text(row: Any, col: str) -> str:
    val = row.get(col, "")
    return "" if val is None else str(val)


def _param_id(kind: str, row: Any, id_cols: list[str], value_col: str | None = None) -> str:
    bits = [kind]
    for c in id_cols:
        v = _row_text(row, c)
        if v and v.lower() != "nan":
            bits.append(f"{_norm(c)}_{_safe_id(v)}")
    if len(bits) == 1 and value_col:
        bits.append(_norm(value_col))
    return _safe_id("__".join(bits))


def _find_col(cols: list[str], accepted: list[str]) -> str | None:
    by_norm = {_norm(c): c for c in cols}
    for a in accepted:
        if _norm(a) in by_norm:
            return by_norm[_norm(a)]
    return None


def _numeric_records_from_json_like(obj: Any, path: Path, family: str) -> list[ParameterRecord]:
    records: list[ParameterRecord] = []
    def walk(prefix: str, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                walk(f"{prefix}_{k}" if prefix else str(k), v)
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                walk(f"{prefix}_{i}", v)
        else:
            f = _as_float(val)
            if f is not None and prefix and source_is_parameter_like(prefix) and not source_is_non_parameter(prefix):
                records.append(ParameterRecord(_safe_id(prefix), f, str(path), None, "strict serialized parameter source", {"path": prefix}))
    walk("", obj)
    return records



def _numeric_columns(df: Any, exclude: set[str]) -> list[str]:
    cols = []
    for c in [str(x) for x in df.columns]:
        if _norm(c) in exclude:
            continue
        if df[c].map(_as_float).notna().any():
            cols.append(c)
    return cols


def _records_from_value_columns(df: Any, path: Path, sheet: str | None, schema: str, kind: str, value_cols: list[str], id_cols: list[str]) -> list[ParameterRecord]:
    records: list[ParameterRecord] = []
    for idx, row in df.iterrows():
        for c in value_cols:
            val = _as_float(row.get(c))
            if val is not None:
                rid = _param_id(kind if len(value_cols) == 1 else c, row, id_cols, c)
                if "sol_id" not in [_norm(x) for x in id_cols] and any(_norm(x) == "time" for x in id_cols):
                    rid += f"__row_{idx}"
                records.append(ParameterRecord(rid, val, str(path), sheet, schema, {k: row.get(k) for k in [*id_cols, c]}))
    return records

def parse_parameter_dataframe(df: Any, path: Path, sheet: str | None, family: str) -> tuple[list[ParameterRecord], str | None, str]:
    cols = [str(c) for c in df.columns]
    norm_cols = {_norm(c): c for c in cols}
    source = sheet or path.stem
    source_norm = _norm(source)
    records: list[ParameterRecord] = []

    if source_is_non_parameter(source):
        return [], "trajectory/prediction/metric/residual/diagnostic table, not a fitted parameter table", "non-parameter"


    filename_norm = _norm(path.name)
    stem_norm = _norm(path.stem)

    # Schema source of truth:
    # - networkmodel/runner.py writes fitted_params_picked.json and model parameter CSVs.
    # - networkmodel/export.py::export_pareto_front_to_excel writes equivalent named physical parameters.
    if family == "networkmodel" and stem_norm in {"model_parameters_genes", "model_parameters_genes_psites", "model_parameters_kinases"}:
        id_cols = [c for c in cols if _norm(c) in {"protein", "gene", "geneid", "psite", "phosphosite", "kinase", "param", "parameter", "name", "sol_id"}]
        value_cols = _numeric_columns(df, NEVER_PARAMETER_COLUMNS | {"time", "pred_fc", "fc", "obs_fc"})
        # Prefer canonical long value column if present; otherwise accept named parameter columns in these exact files.
        if "value" in norm_cols:
            value_cols = [norm_cols["value"]]
        elif "c_k" in norm_cols:
            value_cols = [norm_cols["c_k"]]
        records = _records_from_value_columns(df, path, sheet, f"networkmodel {path.name} parameter table", stem_norm, value_cols, [c for c in id_cols if c not in value_cols])
        return records, None if records else f"{path.name} had no numeric fitted parameter values", f"networkmodel {path.name} parameter table"

    # Schema source of truth:
    # - networkmodel/export.py::save_s_rates_csv / save S rates outputs use protein, psite, time, S columns.
    if family == "networkmodel" and stem_norm == "s_rates_picked":
        value_col = norm_cols.get("s") or norm_cols.get("s_rate") or norm_cols.get("rate")
        if value_col:
            id_cols = [c for c in cols if c != value_col and _norm(c) not in {"pred_fc", "fc"}]
            records = _records_from_value_columns(df, path, sheet, "networkmodel selected S/rate table", "S_rate", [value_col], id_cols)
            return records, None if records else "S_rates_picked had no numeric S/rate values", "networkmodel selected S/rate table"

    # Schema source of truth:
    # - networkmodel/runner.py selected initial state export initial_conditions_y0.csv.
    # Initial conditions are model state metadata; include them separately as dimensionless initial-value parameters.
    if family == "networkmodel" and stem_norm == "initial_conditions_y0":
        value_cols = [norm_cols[c] for c in ("y0", "initial_value", "value") if c in norm_cols]
        if not value_cols:
            value_cols = _numeric_columns(df, NEVER_PARAMETER_COLUMNS - {"id", "index"})[:1]
        id_cols = [c for c in cols if c not in value_cols]
        records = _records_from_value_columns(df, path, sheet, "networkmodel initial_conditions_y0 state table", "initial_condition", value_cols, id_cols)
        return records, None if records else "initial_conditions_y0 had no numeric initial values", "networkmodel initial_conditions_y0 state table"

    # Exact alpha schemas: KinOpt Alpha Values = Gene/Psite/Kinase/Alpha;
    # TFOpt Alpha Values = mRNA/TF/Value (app/tfopt.py loader normalization).
    alpha_col = _find_col(cols, ALPHA_VALUE_COLUMNS + (["value"] if "alpha" in source_norm else []))
    if "alpha" in source_norm and alpha_col:
        id_cols = [norm_cols[n] for n in [_norm(c) for c in ALPHA_ID_COLUMNS] if n in norm_cols and norm_cols[n] != alpha_col]
        if id_cols:
            for _, row in df.iterrows():
                val = _as_float(row.get(alpha_col))
                if val is not None:
                    records.append(ParameterRecord(_param_id("alpha", row, id_cols, alpha_col), val, str(path), sheet, "alpha table from KinOpt/TFOpt export format", {c: row.get(c) for c in [*id_cols, alpha_col]}))
            return records, None if records else "alpha table had no numeric alpha values", "alpha table from KinOpt/TFOpt export format"

    # Exact beta schemas: KinOpt Beta Values = Kinase/Psite/Beta;
    # TFOpt Beta Values = TF/PSite/Value.
    beta_col = _find_col(cols, BETA_VALUE_COLUMNS + (["value"] if "beta" in source_norm else []))
    if "beta" in source_norm and beta_col:
        id_cols = [norm_cols[n] for n in [_norm(c) for c in BETA_ID_COLUMNS] if n in norm_cols and norm_cols[n] != beta_col]
        if id_cols:
            for _, row in df.iterrows():
                val = _as_float(row.get(beta_col))
                if val is not None:
                    records.append(ParameterRecord(_param_id("beta", row, id_cols, beta_col), val, str(path), sheet, "beta table from KinOpt/TFOpt export format", {c: row.get(c) for c in [*id_cols, beta_col]}))
            return records, None if records else "beta table had no numeric beta values", "beta table from KinOpt/TFOpt export format"

    # Networkmodel exact long parameter sheets.
    if source_norm == "params_genes" and {"sol_id", "protein", "param", "value"}.issubset(norm_cols):
        for _, row in df.iterrows():
            val = _as_float(row.get(norm_cols["value"]))
            if val is not None:
                records.append(ParameterRecord(_param_id(str(row.get(norm_cols["param"])), row, [norm_cols["sol_id"], norm_cols["protein"]]), val, str(path), sheet, "networkmodel params_genes sheet", {c: row.get(c) for c in cols}))
        return records, None if records else "params_genes had no numeric values", "networkmodel params_genes sheet"
    if source_norm == "params_kinases" and {"sol_id", "kinase", "c_k"}.issubset(norm_cols):
        for _, row in df.iterrows():
            val = _as_float(row.get(norm_cols["c_k"]))
            if val is not None:
                records.append(ParameterRecord(_param_id("c_k", row, [norm_cols["sol_id"], norm_cols["kinase"]]), val, str(path), sheet, "networkmodel params_kinases sheet", {c: row.get(c) for c in cols}))
        return records, None if records else "params_kinases had no numeric c_k values", "networkmodel params_kinases sheet"
    if source_norm == "deg_sites" and {"sol_id", "protein", "psite", "k_deg"}.issubset(norm_cols):
        for _, row in df.iterrows():
            val = _as_float(row.get(norm_cols["k_deg"]))
            if val is not None:
                records.append(ParameterRecord(_param_id("k_deg", row, [norm_cols["sol_id"], norm_cols["protein"], norm_cols["psite"]]), val, str(path), sheet, "networkmodel deg_sites sheet", {c: row.get(c) for c in cols}))
        return records, None if records else "deg_sites had no numeric k_deg values", "networkmodel deg_sites sheet"

    # Strict long parameter table.
    name_col = _find_col(cols, PARAMETER_NAME_COLUMNS)
    value_col = _find_col(cols, VALUE_COLUMNS)
    if name_col and value_col and source_is_parameter_like(source):
        id_cols = [c for c in cols if c not in {name_col, value_col} and _norm(c) not in NEVER_PARAMETER_COLUMNS]
        for _, row in df.iterrows():
            val = _as_float(row.get(value_col))
            pname = _row_text(row, name_col)
            if val is not None and pname:
                records.append(ParameterRecord(_param_id(pname, row, id_cols, value_col), val, str(path), sheet, "strict long parameter table", {c: row.get(c) for c in [*id_cols, name_col, value_col]}))
        return records, None if records else "long parameter table had no numeric values", "strict long parameter table"

    # ProtWise exact wide parameter sheets: <gene>_params or <gene>_parameters.xlsx.
    protwise_like = source_norm.endswith("_params") or path.stem.lower().endswith("_parameters")
    if protwise_like:
        numeric_cols = []
        for c in cols:
            cn = _norm(c)
            if cn in NEVER_PARAMETER_COLUMNS or cn in {"gene"}:
                continue
            if df[c].map(_as_float).notna().any():
                numeric_cols.append(c)
        if numeric_cols:
            row = df.iloc[-1]
            gene = _row_text(row, norm_cols.get("gene", "")) if "gene" in norm_cols else path.stem.replace("_parameters", "")
            for c in numeric_cols:
                val = _as_float(row.get(c))
                if val is not None:
                    records.append(ParameterRecord(_safe_id(f"{c}__gene_{gene}" if gene else str(c)), val, str(path), sheet, "ProtWise parameter sheet final row", {"Gene": gene, c: row.get(c)}))
            return records, None, "ProtWise parameter sheet final row"

    # Strict wide fallback only for parameter-like sources, never for *_results.
    if source_is_parameter_like(source):
        numeric_cols = []
        for c in cols:
            cn = _norm(c)
            if cn in NEVER_PARAMETER_COLUMNS:
                continue
            if any(cn.startswith(_norm(p)) for p in WIDE_PARAMETER_PREFIXES) and df[c].map(_as_float).notna().any():
                numeric_cols.append(c)
        if numeric_cols:
            id_cols = [c for c in cols if c not in numeric_cols and _norm(c) not in NEVER_PARAMETER_COLUMNS][:4]
            for idx, row in df.iterrows():
                for c in numeric_cols:
                    val = _as_float(row.get(c))
                    if val is not None:
                        records.append(ParameterRecord(_param_id(c, row, id_cols, c) + f"__row_{idx}", val, str(path), sheet, "strict wide parameter table", {k: row.get(k) for k in [*id_cols, c]}))
            return records, None, "strict wide parameter table"

    return [], "unrecognized or non-parameter table; pass --model-family only for family inference, not for metric/trajectory parsing", "unrecognized"


def load_model_result(path: Path, requested_model_family: str | None = None, verbose: bool = False) -> ModelResult | None:
    family, family_source, inferred = assign_model_family(path, requested_model_family)
    if family is None:
        logging.info("Skipped %s because inferred family %s conflicts with --model-family %s", path, inferred, requested_model_family)
        return None
    warnings: list[str] = []
    records: list[ParameterRecord] = []
    skipped: list[SkippedTable] = []
    try:
        suf = path.suffix.lower()
        if suf in {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}:
            for sheet, df in _read_dataframe_file(path):
                recs, reason, schema = parse_parameter_dataframe(df, path, sheet, family)
                if recs:
                    records.extend(recs)
                    if verbose:
                        logging.info("Parsed parameter table:\n- file: %s\n- sheet: %s\n- schema: %s\n- model_family: %s\n- family_source: %s\n- parameters: %d", path, sheet, schema, family, family_source, len(recs))
                else:
                    skipped.append(SkippedTable(str(path), sheet, reason or "no fitted parameters detected"))
                    if verbose:
                        logging.info("Skipped non-parameter table:\n- file: %s\n- sheet: %s\n- reason: %s", path, sheet, reason)
        elif suf == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if family == "networkmodel" and path.name == "fitted_params_picked.json":
                # Schema source of truth: networkmodel selected fitted-parameter JSON named fitted_params_picked.json.
                def walk(prefix, val):
                    if isinstance(val, dict):
                        for k, v in val.items(): walk(f"{prefix}_{k}" if prefix else str(k), v)
                    elif isinstance(val, (list, tuple)):
                        for i, v in enumerate(val): walk(f"{prefix}_{i}", v)
                    else:
                        f = _as_float(val)
                        if f is not None and prefix:
                            records.append(ParameterRecord(_safe_id(prefix), f, str(path), None, "networkmodel fitted_params_picked.json", {"path": prefix}))
                walk("", payload)
            else:
                records = _numeric_records_from_json_like(payload, path, family)
        elif suf == ".tex":
            skipped.append(SkippedTable(str(path), None, "LaTeX equation/model file used for documentation grounding, not a parameter table"))
        elif suf in {".pkl", ".pickle"}:
            # Do not parse arbitrary dashboard/result pickles as parameters except strict parameter-like names.
            if source_is_parameter_like(path.stem) and not source_is_non_parameter(path.stem):
                records = _numeric_records_from_json_like(pickle.loads(path.read_bytes()), path, family)
            else:
                skipped.append(SkippedTable(str(path), None, "pickle artifact is not a named parameter table"))
        elif suf in {".npz", ".npy"}:
            # Raw arrays such as pareto_X.npy lack stable parameter names; skip unless filename is explicitly parameter-like.
            if source_is_parameter_like(path.stem) and not source_is_non_parameter(path.stem):
                import numpy as np
                obj = np.load(path, allow_pickle=True)
                data = {k: obj[k].tolist() for k in obj.files} if hasattr(obj, "files") else obj.tolist()
                records = _numeric_records_from_json_like(data, path, family)
            else:
                skipped.append(SkippedTable(str(path), None, "array artifact lacks project-exported parameter names"))
    except Exception as exc:
        warnings.append(f"Could not load {path}: {exc}")
    if family == "unknown" and records:
        warnings.append(f"Model family could not be inferred for {path}; kept under family 'unknown'. Pass --model-family for family-specific export names/equations.")
    params = {r.parameter_id: r.value for r in records}
    if not records and not skipped:
        skipped.append(SkippedTable(str(path), None, "no fitted parameter schema matched"))
    eq_funcs = EQUATION_SOURCES.get(family, [])
    eq_files = [Path(x.split("::", 1)[0]) for x in eq_funcs if "::" in x]
    return ModelResult(
        family, path, params,
        {"result_file": str(path), "result_files": [str(path)], "schemas": sorted({r.schema for r in records}), "family_source": family_source},
        EQUATIONS.get(family, EQUATIONS["unknown"]), warnings, records, skipped,
        equation_source_files=eq_files,
        equation_source_functions=eq_funcs,
        parameter_source_files=[Path(r.source_file) for r in records],
        schema_source_functions=SCHEMA_SOURCES,
        structure_files=[],
    )


def discover_model_outputs(results_dir: Path, model_family: str | None = None, run_id: str | None = None, verbose: bool = False) -> list[ModelResult]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    root = results_dir / run_id if run_id else results_dir
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    loaded = [r for p in files if (r := load_model_result(p, model_family, verbose)) is not None]
    merged: dict[str, ModelResult] = {}
    for r in loaded:
        if not r.parameters and not r.skipped_tables:
            continue
        m = merged.setdefault(r.model_family, ModelResult(r.model_family, r.source_path, {}, {"result_files": [], "schemas": [], "family_source": r.metadata.get("family_source")}, EQUATIONS.get(r.model_family, EQUATIONS["unknown"]), [], [], [], [], [], [], []))
        m.metadata["result_files"].extend(r.metadata.get("result_files", [str(r.source_path)]))
        m.metadata["schemas"] = sorted(set(m.metadata.get("schemas", [])) | set(r.metadata.get("schemas", [])))
        m.parameters.update(r.parameters)
        m.parameter_records.extend(r.parameter_records)
        m.skipped_tables.extend(r.skipped_tables)
        m.warnings.extend(r.warnings)
        m.equation_source_files = sorted(set(m.equation_source_files + r.equation_source_files), key=str)
        m.equation_source_functions = sorted(set(m.equation_source_functions + r.equation_source_functions))
        m.parameter_source_files = sorted(set(m.parameter_source_files + r.parameter_source_files), key=str)
        m.schema_source_functions = sorted(set(m.schema_source_functions + r.schema_source_functions))
    # Attach existing ProtWise LaTeX equation files and network structure metadata discovered in the run directory.
    for fam, m in merged.items():
        for tex in root.rglob("*_model_latex.tex"):
            if fam in {"protwise", "unknown"}:
                m.equation_source_files.append(tex)
                m.metadata.setdefault("equation_files", []).append(str(tex))
                try:
                    text = tex.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        m.equations.append(text[:2000])
                except OSError:
                    pass
        if fam == "networkmodel":
            for name in ("network_W_global.csv", "network_kinase_inputs.csv", "network_tf_mat.csv", "optimized_entities.json", "metadata.json", "mode_metadata.json", "config_resolved.yaml"):
                f = root / name
                if f.exists():
                    m.structure_files.append(f)
    return list(merged.values())


def build_sbml_document(model_result: ModelResult):
    try:
        import libsbml  # type: ignore
    except Exception:
        return None
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel(); model.setId(_safe_id(model_result.model_family + "_model")); model.setName(model_result.model_family)
    model.setTimeUnits("dimensionless"); model.setExtentUnits("dimensionless"); model.setSubstanceUnits("dimensionless")
    comp = model.createCompartment(); comp.setId("unitless_compartment"); comp.setConstant(True); comp.setSize(1.0); comp.setSpatialDimensions(0); comp.setUnits("dimensionless")
    grounding = "; ".join(model_result.equation_source_functions + model_result.schema_source_functions[:4])
    notes = f"""<body xmlns='http://www.w3.org/1999/xhtml'><p>This SBML model was exported from fitted phoskintime-family results. The fitted observables are fold-change or otherwise unitless measurements. Species and parameters corresponding to these observables are represented as dimensionless quantities. The model is a mathematical regulatory/phosphorylation model, not an absolute concentration-based biochemical model unless additional calibration information is supplied. Time is represented using a dimensionless convention because source result units may be unknown.</p><p>Implementation grounding: {escape(grounding)}</p><p>SBML mapping note: implemented regulatory or ODE equations are represented by dimensionless parameters and a placeholder assignment rule where exact repository-specific simulation state reconstruction is unavailable from result tables alone.</p></body>"""
    model.setNotes(notes)
    obs = model.createSpecies(); obs.setId("unitless_observable"); obs.setCompartment("unitless_compartment"); obs.setInitialAmount(0.0); obs.setSubstanceUnits("dimensionless"); obs.setBoundaryCondition(False); obs.setHasOnlySubstanceUnits(False); obs.setConstant(False)
    for record in sorted(model_result.parameter_records, key=lambda r: r.parameter_id):
        p = model.createParameter(); p.setId(_safe_id(record.parameter_id)); p.setValue(float(record.value)); p.setUnits("dimensionless"); p.setConstant(True)
        p.setNotes(f"<body xmlns='http://www.w3.org/1999/xhtml'><p>Source file: {escape(record.source_file)}; sheet: {escape(str(record.source_sheet))}; schema: {escape(record.schema)}.</p></body>")
    if model_result.equations:
        ar = model.createAssignmentRule(); ar.setVariable("unitless_observable"); ar.setMath(libsbml.parseL3Formula("0")); ar.setNotes(f"<body xmlns='http://www.w3.org/1999/xhtml'><p>Source equation summary: {escape('; '.join(model_result.equations))}</p></body>")
    return doc


def _write_fallback_sbml(model_result: ModelResult, output_path: Path) -> None:
    ns = "http://www.sbml.org/sbml/level3/version2/core"; ET.register_namespace("", ns)
    sbml = ET.Element(f"{{{ns}}}sbml", {"level": "3", "version": "2"})
    model = ET.SubElement(sbml, f"{{{ns}}}model", {"id": _safe_id(model_result.model_family + "_model"), "substanceUnits": "dimensionless", "timeUnits": "dimensionless", "extentUnits": "dimensionless"})
    notes = ET.SubElement(model, f"{{{ns}}}notes"); body = ET.SubElement(notes, "body", {"xmlns": "http://www.w3.org/1999/xhtml"}); ET.SubElement(body, "p").text = "Fold-change/unitless fitted observables are represented with dimensionless units; this is a mathematical regulatory/phosphorylation model, not an absolute concentration model."
    lop = ET.SubElement(model, f"{{{ns}}}listOfParameters")
    for r in sorted(model_result.parameter_records, key=lambda r: r.parameter_id):
        ET.SubElement(lop, f"{{{ns}}}parameter", {"id": _safe_id(r.parameter_id), "value": repr(float(r.value)), "units": "dimensionless", "constant": "true"})
    ET.ElementTree(sbml).write(output_path, encoding="utf-8", xml_declaration=True)


def write_sbml(document, model_result: ModelResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if document is None:
        _write_fallback_sbml(model_result, output_path)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path); ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model-family", choices=FAMILIES); ap.add_argument("--run-id"); ap.add_argument("--validate", action="store_true"); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    if a.verbose:
        logging.info("Source files inspected: %s", sorted({p for paths in SOURCE_HINTS.values() for p in paths}))
        logging.info("Project schema sources: %s", SCHEMA_SOURCES)
    results = discover_model_outputs(a.results_dir, a.model_family, a.run_id, a.verbose)
    exported=[]; reports=[]; found={r.model_family for r in results if r.parameters}
    for fam in FAMILIES:
        if a.model_family and fam != a.model_family: continue
        if fam not in found: logging.warning("%s: skipped, no fitted parameter result files found", fam)
    for r in results:
        if not r.parameters:
            for s in r.skipped_tables[:10]: logging.warning("Skipped non-parameter table: %s sheet=%s reason=%s", s.file, s.sheet, s.reason)
            continue
        out = a.output_dir / f"{r.model_family}_model.xml"
        if out.exists() and not a.overwrite:
            logging.warning("%s exists; use --overwrite to replace", out); continue
        write_sbml(build_sbml_document(r), r, out); exported.append(out)
        rep = validate_sbml(out) if a.validate else ValidationReport(out, 0, 0, ["Validation not requested; pass --validate."]); reports.append(rep)
        print(f"{out}: {rep.errors} errors, {rep.warnings} warnings"); [print(f"  - {m}") for m in rep.messages[:20]]
    print("Fixed implementation grounding:")
    print("- inspected networkmodel/protwise/common/KinOpt/TFOpt equation and export functions")
    print("- recorded schema and equation source files/functions")
    print("- imported existing functions where safe, otherwise mirrored schemas from implementation")
    print("Fixed result parsing:")
    print("- supported results_network_combinatorial layout")
    print("- supported old_results/results_kinopt/kinopt/kinopt_results.xlsx")
    print("- supported old_results/results_tfopt/tfopt/tfopt_results.xlsx")
    print("- supported old_results/results_model/Distributive_results protwise layout")
    print("- removed generic numeric parsing from *_results.* files")
    print("- added workbook sheet-level filtering")
    print("Fixed model-family handling:")
    print("- --model-family is now authoritative when paths are generic")
    print("- valid alpha/beta/parameter tables are not silently dropped")
    print("Outputs:")
    print("- SBML files written and validated where possible")
    print("Discovered model families:"); [print(f"- {fam}: {'exported SBML' if any(p.name.startswith(fam) for p in exported) else 'skipped'}") for fam in [*FAMILIES, 'unknown'] if (not a.model_family or fam == a.model_family) and (fam in found or fam in FAMILIES)]
    print("Outputs:"); [print(f"- {p}") for p in exported]
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
