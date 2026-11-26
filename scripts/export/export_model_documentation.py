#!/usr/bin/env python
"""
Export PhosKinTime model documentation in Markdown, LaTeX and PDF.

This script is read-only with respect to the input result directory. It
grounds equations and result schemas in the repository implementation and
does not infer equations from filenames alone.

For networkmodel, the equations documented here are taken from
networkmodel/backend.py::make_networkmodel_rhs and the loss mapping is
taken from networkmodel/backend.py::multimodal_loss_from_trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from export_model_to_sbml import (
        build_network_genes,
        detect_protwise_model,
        find_col,
        first_existing,
        get_gene_column,
        get_psite_column,
        infer_model_family,
        infer_network_model_id,
        parse_kinopt_xlsx,
        parse_networkmodel_tables,
        parse_protwise_parameters,
        parse_tfopt_xlsx,
        resolve_family,
        safe_id,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "export_model_documentation.py must be placed next to export_model_to_sbml.py "
        "or run from the scripts directory."
    ) from exc


def table_md(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    """Return a Markdown table.

    By default the full table is printed. Pass max_rows only for secondary
    diagnostic tables such as identifier mappings.
    """
    if df is None or df.empty:
        return "_No rows._"
    shown = df if max_rows is None else df.head(max_rows)
    text = shown.to_markdown(index=False)
    if max_rows is not None and len(df) > max_rows:
        text += f"\n\n... ({len(df) - max_rows} more rows)"
    return text


def fmt_number(value: Any) -> str:
    """Compact, stable formatting for fitted numeric values in documentation."""
    try:
        x = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.8g}"


def label(prefix: str, *parts: Any) -> str:
    """Create a readable SBML-safe label used consistently in equations."""
    joined = "_".join([prefix] + [str(p) for p in parts if str(p).strip()])
    return safe_id(joined, prefix.lower())


def code_label(prefix: str, *parts: Any) -> str:
    return f"`{label(prefix, *parts)}`"


def plain_label(prefix: str, *parts: Any) -> str:
    return label(prefix, *parts)


def equation_block(lines: list[str]) -> str:
    return "```text\n" + "\n".join(lines) + "\n```"


def add_term(terms: list[str], coefficient: str, symbol: str, sign: str = "+") -> None:
    coeff = str(coefficient).strip()
    sym = str(symbol).strip()
    if coeff in {"", "1", "+1", "1.0"}:
        body = sym
    elif coeff in {"-1", "-1.0"}:
        body = f"-{sym}"
    else:
        body = f"{coeff}*{sym}"
    if sign == "-" and not body.startswith("-"):
        body = f"-{body}"
    terms.append(body)


def join_terms(terms: list[str]) -> str:
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            expr += " - " + term[1:]
        else:
            expr += " + " + term
    return expr


def read_labeled_matrix(results_dir: Path, filename: str) -> Optional[dict[str, Any]]:
    """Read optional network matrices with row labels in the first column when present."""
    path = results_dir / filename
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    first = df.columns[0]
    first_name = str(first).strip().lower()
    first_is_label = (
            first_name.startswith("unnamed")
            or first_name in {"gene", "protein", "target", "source", "site", "psite", "index", "id"}
            or not pd.api.types.is_numeric_dtype(df[first])
    )
    if first_is_label:
        rows = [str(x).strip() for x in df[first].tolist()]
        data = df.drop(columns=[first])
    else:
        rows = []
        data = df.copy()
    numeric_cols = []
    for c in data.columns:
        vals = pd.to_numeric(data[c], errors="coerce")
        if vals.notna().any():
            numeric_cols.append(c)
            data[c] = vals.fillna(0.0)
    if not numeric_cols:
        return None
    data = data[numeric_cols]
    return {"path": path, "rows": rows, "cols": [str(c).strip() for c in numeric_cols],
            "data": data.reset_index(drop=True)}


def matrix_row_index(matrix: Optional[dict[str, Any]], target: str, fallback_index: Optional[int] = None) -> Optional[
    int]:
    if matrix is None:
        return None
    rows = matrix.get("rows") or []
    if rows:
        target_clean = str(target).strip()
        lower = {r.lower(): i for i, r in enumerate(rows)}
        if target_clean.lower() in lower:
            return lower[target_clean.lower()]
        # Relaxed matching for rows such as "ABL2|Y245" or "ABL2_Y245".
        for i, r in enumerate(rows):
            if target_clean.lower() in r.lower():
                return i
        return None
    if fallback_index is not None and 0 <= fallback_index < len(matrix["data"]):
        return fallback_index
    return None


def matrix_terms(matrix: Optional[dict[str, Any]], row_idx: Optional[int], symbol_prefix: str) -> list[str]:
    if matrix is None or row_idx is None:
        return []
    if row_idx < 0 or row_idx >= len(matrix["data"]):
        return []
    row = matrix["data"].iloc[row_idx]
    terms: list[str] = []
    for col in matrix["cols"]:
        try:
            value = float(row[col])
        except Exception:
            continue
        if not math.isfinite(value) or abs(value) <= 1e-12:
            continue
        add_term(terms, fmt_number(value), f"{plain_label(symbol_prefix, col)}(t)")
    return terms


def site_matrix_row_index(matrix: Optional[dict[str, Any]], gene: str, site: str,
                          fallback_index: Optional[int] = None) -> Optional[int]:
    if matrix is None:
        return None
    rows = matrix.get("rows") or []
    if rows:
        g = str(gene).strip().lower()
        s = str(site).strip().lower()
        both = [(i, r.lower()) for i, r in enumerate(rows)]
        for i, r in both:
            if g in r and s in r:
                return i
        for i, r in both:
            if s == r or s in r:
                return i
        return None
    return fallback_index if fallback_index is not None else None


def tf_input_equation(gene_name: str, gene_index: int, tf_matrix: Optional[dict[str, Any]]) -> str:
    t_id = plain_label("T", gene_name)
    row_idx = matrix_row_index(tf_matrix, gene_name, gene_index)
    terms = matrix_terms(tf_matrix, row_idx, "Q")
    denom = f"max({plain_label('tfdeg', gene_name)}, 1e-12)"
    if terms:
        return f"{t_id}(t) = ({join_terms(terms)})/{denom}"
    return f"{t_id}(t) = TF_input_{safe_id(gene_name, 'gene')}(t)  # no explicit TF matrix row found in result directory"


def s_drive_equation(gene_name: str, site: str, site_index: int, s_value: float,
                     w_matrix: Optional[dict[str, Any]]) -> str:
    s_id = plain_label("S", gene_name, site)
    row_idx = site_matrix_row_index(w_matrix, gene_name, site, site_index)
    terms = matrix_terms(w_matrix, row_idx, "K")
    if terms:
        return f"{s_id}(t) = {join_terms(terms)}"
    return f"{s_id}(t) = {fmt_number(s_value)}"


def shared_synthesis_lines(gene: str) -> list[str]:
    """Return labelled TF-regulated synthesis equations used by all networkmodel topologies."""
    R = plain_label("R", gene)
    A = plain_label("A", gene)
    B = plain_label("B", gene)
    T = plain_label("T", gene)
    u = plain_label("u", gene)
    synth = plain_label("synth", gene)
    return [
        f"{u}(t) = {T}(t)/(1 + abs({T}(t)))",
        f"{synth}(t) = {A}*(1 + {plain_label('tau', gene)}*{u}(t)/(1 + {u}(t) + 1e-6))  if {u}(t) >= 0",
        f"{synth}(t) = {A}/(1 + {plain_label('tau', gene)}*abs({u}(t)))                    if {u}(t) < 0",
        f"d{R}/dt = {synth}(t) - {B}*{R}(t)",
    ]


def explicit_distributive_gene_equations(g: Any, *, saturated: bool = False) -> str:
    """Expand MODEL=0 distributive equations, or MODEL=4 saturated distributive-layout equations."""
    gene = g.name
    R = plain_label("R", gene)
    P = plain_label("P", gene)
    C = plain_label("C", gene)
    D = plain_label("D", gene)
    E = plain_label("E", gene)
    lines: list[str] = shared_synthesis_lines(gene)
    fwd_terms: list[str] = []
    back_terms: list[str] = []
    trans = f"({C}*{R}(t))/(1 + {R}(t))" if saturated else f"{C}*{R}(t)"
    for site in g.sites:
        X = plain_label("X", gene, site)
        S = plain_label("S", gene, site)
        Dp = plain_label("Dp", gene, site)
        F = f"({S}(t)*{P}(t))/(1 + {P}(t))" if saturated else f"{S}(t)*{P}(t)"
        fwd_terms.append(F)
        back_terms.append(f"{E}*{X}(t)")
        if saturated:
            lines.append(f"d{X}/dt = {F} - ({Dp} + {D})*{X}(t) - {E}*{X}(t)")
        else:
            lines.append(f"d{X}/dt = {F} - ({E} + {Dp} + {D})*{X}(t)")
    p_terms = [trans, f"-{D}*{P}(t)"]
    for term in fwd_terms:
        p_terms.append(f"-{term}")
    for term in back_terms:
        p_terms.append(term)
    protein_line = f"d{P}/dt = {join_terms(p_terms)}"
    lines.insert(4, protein_line)
    return equation_block(lines)


def explicit_successive_gene_equations(g: Any) -> str:
    """Expand MODEL=1 sequential/successive equations from networkmodel.models.sequential_rhs."""
    gene = g.name
    sites = list(g.sites)
    R = plain_label("R", gene)
    P0 = plain_label("P", gene)
    C = plain_label("C", gene)
    D = plain_label("D", gene)
    E = plain_label("E", gene)
    lines: list[str] = shared_synthesis_lines(gene)
    if not sites:
        lines.append(f"d{P0}/dt = {C}*{R}(t) - {D}*{P0}(t)")
        return equation_block(lines)

    first_site = sites[0]
    S1 = plain_label("S", gene, first_site)
    X1 = plain_label("X", gene, first_site)
    lines.append(f"d{P0}/dt = {C}*{R}(t) - {D}*{P0}(t) - {S1}(t)*{P0}(t) + {E}*{X1}(t)")

    if len(sites) == 1:
        Dp1 = plain_label("Dp", gene, first_site)
        lines.append(f"d{X1}/dt = {S1}(t)*{P0}(t) - ({E} + {Dp1} + {D})*{X1}(t)")
        return equation_block(lines)

    # First ordered phospho state: P0 -> X1 -> X2.
    second_site = sites[1]
    X2 = plain_label("X", gene, second_site)
    S2 = plain_label("S", gene, second_site)
    Dp1 = plain_label("Dp", gene, first_site)
    lines.append(f"d{X1}/dt = {S1}(t)*{P0}(t) + {E}*{X2}(t) - ({S2}(t) + {E} + {Dp1} + {D})*{X1}(t)")

    # Middle ordered phospho states.
    for j in range(1, len(sites) - 1):
        site = sites[j]
        prev_site = sites[j - 1]
        next_site = sites[j + 1]
        Xj = plain_label("X", gene, site)
        Xprev = plain_label("X", gene, prev_site)
        Xnext = plain_label("X", gene, next_site)
        Sj = plain_label("S", gene, site)
        Snext = plain_label("S", gene, next_site)
        Dpj = plain_label("Dp", gene, site)
        lines.append(f"d{Xj}/dt = {Sj}(t)*{Xprev}(t) + {E}*{Xnext}(t) - ({Snext}(t) + {E} + {Dpj} + {D})*{Xj}(t)")

    # Last ordered phospho state.
    last_site = sites[-1]
    prev_site = sites[-2]
    Xlast = plain_label("X", gene, last_site)
    Xprev = plain_label("X", gene, prev_site)
    Slast = plain_label("S", gene, last_site)
    Dplast = plain_label("Dp", gene, last_site)
    lines.append(f"d{Xlast}/dt = {Slast}(t)*{Xprev}(t) - ({E} + {Dplast} + {D})*{Xlast}(t)")
    return equation_block(lines)


def network_model_topology(model_id: int) -> str:
    return {0: "distributive", 1: "successive/sequential", 2: "combinatorial", 4: "saturated"}.get(model_id,
                                                                                                   "unknown-standard")


def explicit_combinatorial_gene_equations(g: Any) -> str:
    gene = g.name
    sites = list(g.sites)
    n = len(sites)
    R = plain_label("R", gene)
    A = plain_label("A", gene)
    B = plain_label("B", gene)
    C = plain_label("C", gene)
    D = plain_label("D", gene)
    E = plain_label("E", gene)
    T = plain_label("T", gene)
    u = plain_label("u", gene)
    synth = plain_label("synth", gene)
    lines: list[str] = []
    lines.append(f"{u}(t) = {T}(t)/(1 + abs({T}(t)))")
    lines.append(f"{synth}(t) = {A}*(1 + {plain_label('tau', gene)}*{u}(t)/(1 + {u}(t) + 1e-6))  if {u}(t) >= 0")
    lines.append(f"{synth}(t) = {A}/(1 + {plain_label('tau', gene)}*abs({u}(t)))                    if {u}(t) < 0")
    lines.append(f"d{R}/dt = {synth}(t) - {B}*{R}(t)")
    if n == 0:
        P0 = plain_label("P", gene, "mask0")
        lines.append(f"d{P0}/dt = {C}*{R}(t) - {D}*{P0}(t)")
        return equation_block(lines)
    n_masks = 2 ** n
    for m in range(n_masks):
        Pm = plain_label("P", gene, f"mask{m}")
        terms: list[str] = []
        if m == 0:
            terms.append(f"{C}*{R}(t)")
            terms.append(f"-{D}*{Pm}(t)")
        for j, site in enumerate(sites):
            bit = 1 << j
            S = plain_label("S", gene, site)
            Dp = plain_label("Dp", gene, site)
            if m & bit:
                Pclear = plain_label("P", gene, f"mask{m ^ bit}")
                terms.append(f"{S}(t)*{Pclear}(t)")
                terms.append(f"-{E}*{Pm}(t)")
                terms.append(f"-({Dp} + {D})*{Pm}(t)")
            else:
                Pset = plain_label("P", gene, f"mask{m | bit}")
                terms.append(f"{E}*{Pset}(t)")
                terms.append(f"-{S}(t)*{Pm}(t)")
        active_sites = [sites[j] for j in range(n) if m & (1 << j)]
        state_note = "unphosphorylated" if not active_sites else " + ".join(active_sites)
        lines.append(f"d{Pm}/dt = {join_terms(terms)}    # mask {m}: {state_note}")
    return equation_block(lines)


def network_equations_markdown(results_dir: Path, parsed: dict[str, Any], genes: list[Any], model_id: int) -> str:
    """Return every labelled networkmodel ODE, expanded gene by gene and site by site."""
    tf_matrix = read_labeled_matrix(results_dir, "network_tf_mat.csv")
    w_matrix = read_labeled_matrix(results_dir, "network_W_global.csv")
    topology = network_model_topology(model_id)
    saturated = model_id == 4
    combinatorial = model_id == 2
    successive = model_id == 1
    distributive = model_id == 0
    lines: list[str] = []
    lines.append(
        f"The networkmodel equations below are expanded from the implemented backend topology selected by `MODEL={model_id}` ({topology}). `MODEL=0` is distributive, `MODEL=1` is successive/sequential, `MODEL=2` is combinatorial, and `MODEL=4` is the saturated branch. This section intentionally avoids index-only placeholder equations: every gene, site, mask and fitted parameter label is written explicitly.")
    lines.append("")
    lines.append("### Backend construction shared by the expanded equations")
    lines.append("")
    lines.append(
        "For each listed gene, the mRNA equation uses the backend TF-regulated synthesis function. If `network_tf_mat.csv` is present, each `T_<gene>(t)` line is expanded from the non-zero entries in that matrix. If it is absent, the line is kept as a labelled `TF_input_<gene>(t)` placeholder because the result directory does not contain the TF adjacency values.")
    lines.append("")
    lines.append(
        "For each phosphosite, `S_<gene>_<site>(t)` is expanded from `network_W_global.csv` when that matrix is present and row labels can be matched. Otherwise it is written as the fitted/selected value parsed from `S_rates_picked.csv`, or `0` when absent.")
    lines.append("")
    if tf_matrix is not None:
        lines.append(f"TF matrix parsed for labelled TF-input expansion: `{tf_matrix['path']}`")
    else:
        lines.append("TF matrix not found; TF inputs are kept as labelled external inputs.")
    if w_matrix is not None:
        lines.append(f"Network kinase-site matrix parsed for labelled S-rate expansion: `{w_matrix['path']}`")
    else:
        lines.append("Network kinase-site matrix not found; S-rate labels use selected fitted values.")
    lines.append("")
    lines.append("### Explicit gene-by-gene ODEs")
    lines.append("")
    equation_count = 0
    for gi, g in enumerate(genes):
        gene = g.name
        lines.append(f"#### {gene}")
        lines.append("")
        state_items = [f"mRNA state: {code_label('R', gene)}"]
        if combinatorial:
            n_masks = 2 ** len(g.sites) if g.sites else 1
            state_items.append(
                "protein mask states: " + ", ".join(code_label('P', gene, f'mask{m}') for m in range(n_masks)))
        else:
            if successive:
                state_items.append(f"unphosphorylated protein / chain state P0: {code_label('P', gene)}")
                if g.sites:
                    ordered = [f"{code_label('X', gene, site)} = ordered chain state P{k}" for k, site in
                               enumerate(g.sites, start=1)]
                    state_items.append("successive phospho-chain states: " + ", ".join(ordered))
            else:
                state_items.append(f"protein state: {code_label('P', gene)}")
                if g.sites:
                    state_items.append("distributive/saturated phosphosite states: " + ", ".join(
                        code_label('X', gene, site) for site in g.sites))
        lines.extend([f"- {item}" for item in state_items])
        lines.append("")
        param_rows = [
            (plain_label("A", gene), g.A, "mRNA synthesis"),
            (plain_label("B", gene), g.B, "mRNA degradation"),
            (plain_label("C", gene), g.C, "translation"),
            (plain_label("D", gene), g.D, "protein degradation"),
            (plain_label("E", gene), g.E, "de-phosphorylation return"),
            (plain_label("tau", gene), g.tf_scale, "global TF scale applied to this gene"),
        ]
        for pid, value, meaning in param_rows:
            lines.append(f"- `{pid}` = {fmt_number(value)}  ({meaning})")
        for site_idx, site in enumerate(g.sites):
            lines.append(
                f"- `{plain_label('Dp', gene, site)}` = {fmt_number(g.Dp.get(site, 0.0))}  (phospho-state degradation for `{site}`)")
            lines.append(
                f"- `{plain_label('S', gene, site)}` = {fmt_number(g.S.get(site, 0.0))}  (selected site phosphorylation input for `{site}`)")
        lines.append("")
        # Explicit input definitions.
        input_lines = [tf_input_equation(gene, gi, tf_matrix)]
        for site_idx, site in enumerate(g.sites):
            input_lines.append(s_drive_equation(gene, site, site_idx, g.S.get(site, 0.0), w_matrix))
        lines.append("Input labels used by this gene:")
        lines.append(equation_block(input_lines))
        lines.append("")
        if combinatorial:
            lines.append(explicit_combinatorial_gene_equations(g))
            equation_count += 1 + (2 ** len(g.sites) if g.sites else 1)
        elif successive:
            lines.append(explicit_successive_gene_equations(g))
            equation_count += 2 + len(g.sites)
        elif distributive or saturated:
            lines.append(explicit_distributive_gene_equations(g, saturated=saturated))
            equation_count += 2 + len(g.sites)
        else:
            lines.append(explicit_distributive_gene_equations(g, saturated=False))
            equation_count += 2 + len(g.sites)
        lines.append("")
    lines.append("### Explicit observable and loss labels")
    lines.append("")
    obs_lines: list[str] = []
    for g in genes:
        gene = g.name
        R = plain_label("R", gene)
        obs_lines.append(f"obs_mRNA_{safe_id(gene, 'gene')}(t) = {R}(t)/{R}(t0)")
        if combinatorial:
            masks = [plain_label("P", gene, f"mask{m}") for m in range(2 ** len(g.sites) if g.sites else 1)]
            obs_lines.append(
                f"obs_protein_{safe_id(gene, 'gene')}(t) = ({' + '.join(m + '(t)' for m in masks)})/({' + '.join(m + '(t0)' for m in masks)})")
            for j, site in enumerate(g.sites):
                masks_with_site = [plain_label("P", gene, f"mask{m}") for m in range(2 ** len(g.sites)) if m & (1 << j)]
                if masks_with_site:
                    obs_lines.append(
                        f"obs_phospho_{safe_id(gene, 'gene')}_{safe_id(site, 'site')}(t) = ({' + '.join(m + '(t)' for m in masks_with_site)})/({' + '.join(m + '(t0)' for m in masks_with_site)})")
        else:
            states = [plain_label("P", gene)] + [plain_label("X", gene, site) for site in g.sites]
            obs_lines.append(
                f"obs_protein_{safe_id(gene, 'gene')}(t) = ({' + '.join(s + '(t)' for s in states)})/({' + '.join(s + '(t0)' for s in states)})")
            for site in g.sites:
                X = plain_label("X", gene, site)
                obs_lines.append(f"obs_phospho_{safe_id(gene, 'gene')}_{safe_id(site, 'site')}(t) = {X}(t)/{X}(t0)")
    lines.append(equation_block(obs_lines))
    lines.append("")
    if successive:
        lines.append(
            "For `MODEL=1`, `X_<gene>_<site>` labels are ordered successive chain states, not independent distributive site pools. The order is the row/order parsed from `model_parameters_genes_psites.csv`; `S_<gene>_<site_j>` drives the transition from the previous chain state into that site state.")
        lines.append("")
    if distributive:
        lines.append(
            "For `MODEL=0`, every `X_<gene>_<site>` is a distributive site pool fed directly from the unphosphorylated protein state `P_<gene>` and returned by the de-phosphorylation flux `E_<gene>*X_<gene>_<site>`.")
        lines.append("")
    lines.append(
        "The scalar objective is the weighted sum of the available labelled residuals only: `L_total = lambda_mRNA*L_mRNA + lambda_protein*L_protein + lambda_phospho*L_phospho`, with unavailable layers omitted exactly as in `detect_data_mode` and `multimodal_loss_from_trajectory`.")
    lines.append("")
    lines.append(f"Expanded dynamic equation count written in this section: `{equation_count}`.")
    return "\n".join(lines)


def protwise_equations_markdown(latex_files: list[Path]) -> str:
    if latex_files:
        try:
            content = latex_files[0].read_text(errors="ignore")
            return (
                    "Existing LaTeX model equations were found and are used as the preferred equation reference:\n\n"
                    f"`{latex_files[0]}`\n\n"
                    "```tex\n" + content[:6000] + "\n```"
            )
        except Exception:
            pass
    return r"""
The local protwise/distmod equations are grounded in `protwise.models.distmod` and the Diffrax implementation in `protwise/models/diffrax_solver.py`.

Let \(R\) be mRNA, \(P\) unphosphorylated protein, and \(X_i\) phosphorylated site \(i\).

\[
\frac{dR}{dt}=A-BR
\]

\[
\frac{dP}{dt}=CR-\left(D+\sum_i S_i\right)P+\sum_i X_i
\]

\[
\frac{dX_i}{dt}=S_iP-(1+D_i)X_i
\]

Here \(A,B,C,D\) are the fitted base kinetic rates, \(S_i\) are site phosphorylation rates, and \(D_i\) are site dephosphorylation/degradation rates.
"""


def kin_tf_equations_markdown(family: str) -> str:
    if family == "kinopt":
        return (
            "KinOpt exports fitted kinase-regulatory coefficients. The confirmed fitted sheets are `Alpha Values` "
            "with columns `Gene`, `Psite`, `Kinase`, `Alpha`, and `Beta Values` with columns `Kinase`, `Psite`, `Beta`. "
            "The SBML representation contains these coefficients as dimensionless constant parameters. Trajectories, "
            "residuals, summaries and metrics are not exported as parameters."
        )
    return (
        "TFOpt exports fitted transcription-factor regulatory coefficients. The confirmed fitted sheets are `Alpha Values` "
        "with columns `mRNA`, `TF`, `Value`, and `Beta Values` with columns `TF`, `PSite`, `Value`. "
        "The SBML representation contains these coefficients as dimensionless constant parameters. Predictions, "
        "residuals, optimization metrics and summaries are not exported as parameters."
    )


def network_metadata_and_tables(results_dir: Path) -> tuple[dict[str, Any], list[tuple[str, pd.DataFrame]], str]:
    parsed = parse_networkmodel_tables(results_dir)
    genes = build_network_genes(parsed)
    model_id = infer_network_model_id(results_dir)
    gene_df = parsed["gene_df"]
    gene_col = parsed["gene_col"]
    table = gene_df.copy()
    if gene_col != "Gene":
        table = table.rename(columns={gene_col: "Gene"})

    site_rows: list[dict[str, Any]] = []
    for g in genes:
        for site in g.sites:
            site_rows.append({
                "Gene": g.name,
                "Psite": site,
                "S_label": plain_label("S", g.name, site),
                "S_value": g.S.get(site, 0.0),
                "Dp_label": plain_label("Dp", g.name, site),
                "Dp_value": g.Dp.get(site, 0.0),
                "X_state": plain_label("X", g.name, site),
            })
    site_table = pd.DataFrame(site_rows)

    parsed_sources = [str(p) for p in parsed["files_parsed"]]
    metadata = {
        "equation_source_files": ["networkmodel/backend.py"],
        "equation_source_functions": ["make_networkmodel_rhs", "multimodal_loss_from_trajectory",
                                      "_global_networkmodel_observable"],
        "parameter_source_files": [str(p) for p in parsed["files_parsed"] if
                                   Path(p).name.startswith("model_parameters") or Path(p).name == "S_rates_picked.csv"],
        "structure_source_files": [p for p in parsed_sources if
                                   Path(p).name in {"network_tf_mat.csv", "network_W_global.csv",
                                                    "network_kinase_inputs.csv", "initial_conditions_y0.csv",
                                                    "optimized_entities.json", "metadata.json", "mode_metadata.json"}],
        "schema_source_functions": ["networkmodel/backend.py::_unpack_theta_jax",
                                    "networkmodel/backend.py::make_networkmodel_rhs"],
        "skipped_files": [str(p) for p in parsed["files_skipped"]],
        "warnings": parsed.get("warnings", []),
        "networkmodel_model_id": model_id,
        "networkmodel_n_genes": len(genes),
        "networkmodel_n_sites": sum(len(g.sites) for g in genes),
        "networkmodel_topology": network_model_topology(model_id),
        "networkmodel_expanded_equations": sum(
            (1 + (2 ** len(g.sites) if g.sites else 1)) if model_id == 2 else (2 + len(g.sites)) for g in genes),
    }
    tables = [("Networkmodel fitted gene parameters", table)]
    if not site_table.empty:
        tables.append(("Networkmodel labelled site parameters used in explicit equations", site_table))
    return metadata, tables, network_equations_markdown(results_dir, parsed, genes, model_id)


def protwise_model_kind(entry: dict[str, Any]) -> str:
    """Infer protwise model kind from path plus fitted parameter names."""
    path_text = str(entry.get("path", "")).lower()
    if "succ" in path_text or "successive" in path_text or "sequential" in path_text:
        return "succmod"
    if "rand" in path_text or "random" in path_text:
        return "randmod"
    return detect_protwise_model(entry["params"])


def protwise_metadata_and_tables(results_dir: Path) -> tuple[dict[str, Any], list[tuple[str, pd.DataFrame]], str]:
    parsed = parse_protwise_parameters(results_dir)

    rows: list[dict[str, Any]] = []
    for entry in parsed["gene_params"]:
        gene = entry["gene"]
        model = protwise_model_kind(entry)
        for pname, value in entry["params"].items():
            rows.append({
                "Gene": gene,
                "Model": model,
                "Parameter": str(pname),
                "Value": value,
                "Source": str(entry["path"]),
            })

    table = pd.DataFrame(rows)

    metadata = {
        "equation_source_files": [
            "protwise/models/diffrax_solver.py",
            "protwise/models/distmod.py",
            "protwise/models/succmod.py",
            "protwise/models/randmod.py",
        ],
        "equation_source_functions": ["_dist_rhs", "_succ_rhs", "_rand_rhs", "ode_core"],
        "parameter_source_files": [str(p) for p in parsed["files_parsed"]],
        "schema_source_functions": [
            "protwise/paramest/core.py::process_gene",
            "common/utils/display.py::save_result",
        ],
        "skipped_files": [str(p) for p in parsed["files_skipped"]],
        "warnings": parsed.get("warnings", []),
        "protwise_n_proteins": len(parsed["gene_params"]),
        "protwise_table_layout": "long: one fitted parameter per row",
    }

    equations = (
        "Protwise equations are rendered protein-by-protein in the LaTeX/PDF output. "
        "The Markdown table above contains the fitted values used in those equations."
    )

    return metadata, [("Protwise fitted parameters", table)], equations


def kin_tf_metadata_and_tables(results_dir: Path, family: str) -> tuple[
    dict[str, Any], list[tuple[str, pd.DataFrame]], str]:
    wb_name = "kinopt_results.xlsx" if family == "kinopt" else "tfopt_results.xlsx"
    wb = first_existing(results_dir, (wb_name,))
    if wb is None:
        raise FileNotFoundError(f"Could not find {wb_name}")
    alpha, beta = parse_kinopt_xlsx(wb) if family == "kinopt" else parse_tfopt_xlsx(wb)
    metadata = {
        "equation_source_files": ["coefficient-only algebraic model"],
        "equation_source_functions": [],
        "parameter_source_files": [str(wb)],
        "schema_source_functions": [
            "kinopt/local/exporter/sheetutils.py::output_results" if family == "kinopt" else "tfopt/local/exporter/sheetutils.py::save_results_to_excel"
        ],
        "skipped_files": [],
        "warnings": [],
    }
    return metadata, [("Alpha fitted parameters", alpha), ("Beta fitted parameters", beta)], kin_tf_equations_markdown(
        family)


def sbml_mapping_table(tables: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for _, df in tables:
        for col in df.columns:
            rows.append({"original_name": str(col), "sbml_safe_id": safe_id(col)})
    return pd.DataFrame(rows).drop_duplicates()


def build_markdown(results_dir: Path, family: str, metadata: dict[str, Any], tables: list[tuple[str, pd.DataFrame]],
                   equations: str) -> str:
    lines: list[str] = []
    lines.append(f"# Model Documentation ({family})")
    lines.append("")
    lines.append(f"**Result directory:** `{results_dir}`")
    lines.append(f"**Model family:** `{family}`")
    lines.append("")
    lines.append("## Implementation grounding")
    lines.append("")
    lines.append(
        "The documentation is grounded in the implementation files/functions listed in the metadata block. Networkmodel equations are written directly from `networkmodel/backend.py::make_networkmodel_rhs`; no placeholder ODE text is used.")
    lines.append("")
    lines.append("## Source files and functions")
    lines.append("")
    lines.append("**Equation source files:**")
    for p in metadata.get("equation_source_files", []):
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("**Equation source functions:**")
    for f in metadata.get("equation_source_functions", []):
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("**Schema/export source functions:**")
    for f in metadata.get("schema_source_functions", []):
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Result files parsed")
    lines.append("")
    for p in metadata.get("parameter_source_files", []):
        lines.append(f"- `{p}`")
    if not metadata.get("parameter_source_files"):
        lines.append("_None._")
    lines.append("")
    lines.append("## Result files skipped")
    lines.append("")
    for p in metadata.get("skipped_files", []):
        lines.append(f"- `{p}`")
    if not metadata.get("skipped_files"):
        lines.append("_No skipped files were detected in the result directory._")
    lines.append("")
    lines.append("## Fitted parameter tables")
    lines.append("")
    for title, df in tables:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(table_md(df, max_rows=None))
        lines.append("")
    mapping = sbml_mapping_table(tables)
    lines.append("## Original names and SBML-safe identifiers")
    lines.append("")
    lines.append(table_md(mapping, max_rows=80))
    lines.append("")
    lines.append("## Equations")
    lines.append("")
    lines.append(equations.strip())
    lines.append("")
    lines.append("## Term explanations")
    lines.append("")
    lines.append("- `R`/`R_i`: mRNA/transcript state.")
    lines.append(
        "- `P`/`P_i`: unphosphorylated protein state for MODEL 0/1/4; mask state for MODEL 2 when suffixed by `mask*`.")
    lines.append("- `X_i`/`X_ij`: MODEL 0/4 distributive site-level state; MODEL 1 ordered successive chain state.")
    lines.append("- `A`, `B`, `C`, `D`: synthesis, mRNA degradation, translation and protein degradation parameters.")
    lines.append("- `S_i`/`S_ij`: phosphorylation input/rate for site `i` or gene-site pair `(i,j)`.")
    lines.append("- `D_i`/`D^p_ij`: phospho-state dephosphorylation/degradation parameter.")
    lines.append("- `E`: de-phosphorylation return flux parameter.")
    lines.append(
        "- `TF`, `W`, `K`: transcription-factor matrix, kinase-site network matrix and kinase input trajectory/matrix.")
    lines.append("")
    lines.append("## Constraints and assumptions")
    lines.append("")
    lines.append(
        "- All fitted values are exported as dimensionless, because the data are fold-change/unitless measurements.")
    lines.append("- No concentration units or mass-action units are invented.")
    lines.append(
        "- Prediction, residual, trajectory, objective, Pareto, metric, diagnostic, plot and pickle outputs are skipped as fitted parameter sources.")
    lines.append(
        "- For networkmodel SBML, `S_rates_picked.csv` is preferred for finite exported `S_ij` values. If it is absent or incomplete, missing `S_ij` values are exported as zero-valued constants so the file remains explicit and valid.")
    lines.append(
        "- The full backend time-varying kinase interpolation and sparse matrix construction remain documented as provenance. The SBML encodes a finite ODE system with fitted result values available in the result directory.")
    lines.append("")
    lines.append("## SBML mapping notes")
    lines.append("")
    lines.append("- SBML IDs are sanitized versions of the original biological identifiers.")
    lines.append("- Original names are preserved in SBML `name` fields and model notes where possible.")
    lines.append("- Compartments, species and parameters use `dimensionless` units.")
    lines.append(
        "- Regulatory/algebraic quantities are represented with assignment rules; dynamic states use rate rules.")
    lines.append("")
    lines.append("## Internal metadata")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metadata, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def latex_escape(text: Any) -> str:
    """Escape text for normal LaTeX text mode."""
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def latex_math_id(text: Any) -> str:
    """Render a readable model label in math mode."""
    return r"\mathrm{" + latex_escape(str(text)).replace(r"\textbackslash{}", r"\backslash{}") + "}"


def latex_value(value: Any) -> str:
    """Format a table/equation value for LaTeX."""
    try:
        x = float(value)
    except Exception:
        return latex_escape(value)
    if not math.isfinite(x):
        return "nan"
    # Scientific notation converted to \times 10^{} for math mode.
    txt = f"{x:.8g}"
    if "e" in txt or "E" in txt:
        base, exp = txt.lower().split("e")
        return rf"{base}\times 10^{{{int(exp)}}}"
    return txt


def latex_cell(value: Any, numeric: bool = False) -> str:
    if numeric:
        return "$" + latex_value(value) + "$"
    return latex_escape(value)


def latex_column_spec(df: pd.DataFrame, title: str = "") -> str:
    """Compact fixed-width landscape table layout.

    The previous version used natural-width `r` columns for numeric values.
    That lets long numeric-column headers exceed the page width. This version
    forces both text and numeric columns into fixed-width p{} columns.
    """
    title_l = title.lower()
    cols = list(df.columns)

    # Table 1: gene-level kinetic parameters, usually 10 columns.
    if "fitted gene parameters" in title_l:
        specs = []
        for c in cols:
            if str(c).lower() == "gene":
                specs.append(r">{\raggedright\arraybackslash}p{0.095\linewidth}")
            else:
                specs.append(r">{\raggedleft\arraybackslash}p{0.085\linewidth}")
        return "".join(specs)

    # Table 2: labelled site parameters, usually 7 columns.
    if "labelled site parameters" in title_l:
        widths = {
            "Gene": "0.075",
            "Psite": "0.075",
            "S_label": "0.175",
            "S_value": "0.075",
            "Dp_label": "0.175",
            "Dp_value": "0.075",
            "X_state": "0.175",
        }
        specs = []
        for c in cols:
            w = widths.get(str(c), "0.11")
            align = r"\raggedleft" if pd.api.types.is_numeric_dtype(df[c]) else r"\raggedright"
            specs.append(r">{" + align + r"\arraybackslash}p{" + w + r"\linewidth}")
        return "".join(specs)

    # Generic fallback.
    n = max(len(cols), 1)
    width = min(0.18, max(0.06, 0.92 / n))
    specs = []
    for c in cols:
        align = r"\raggedleft" if pd.api.types.is_numeric_dtype(df[c]) else r"\raggedright"
        specs.append(r">{" + align + r"\arraybackslash}p{" + f"{width:.3f}" + r"\linewidth}")
    return "".join(specs)


def latex_longtable(title: str, df: pd.DataFrame) -> str:
    """Full dataframe as a clean landscape longtable."""
    if df is None or df.empty:
        return "\\paragraph{" + latex_escape(title) + "} No rows.\n"

    title_l = title.lower()
    tiny_table = (
            "fitted gene parameters" in title_l
            or "labelled site parameters" in title_l
    )

    out: list[str] = []
    out.append(r"\begin{landscape}")
    out.append(r"\tiny" if tiny_table else r"\scriptsize")
    out.append(r"\setlength{\tabcolsep}{1.6pt}" if tiny_table else r"\setlength{\tabcolsep}{3pt}")
    out.append(r"\renewcommand{\arraystretch}{1.15}")

    spec = latex_column_spec(df, title=title)

    out.append(r"\begin{longtable}{" + spec + "}")
    out.append(r"\caption{" + latex_escape(title) + r"}\\")
    headers = [latex_escape(c) for c in df.columns]
    header_line = " & ".join(headers) + r" \\"
    out.append(r"\toprule")
    out.append(header_line)
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\caption[]{" + latex_escape(title) + r" (continued)}\\")
    out.append(r"\toprule")
    out.append(header_line)
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\midrule")
    out.append(r"\multicolumn{" + str(len(df.columns)) + r"}{r}{Continued on next page}\\")
    out.append(r"\endfoot")
    out.append(r"\bottomrule")
    out.append(r"\endlastfoot")

    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            numeric = pd.api.types.is_numeric_dtype(df[c])
            cells.append(latex_cell(row[c], numeric=numeric))
        out.append(" & ".join(cells) + r" \\")

    out.append(r"\end{longtable}")
    out.append(r"\end{landscape}")
    out.append("")
    return "\n".join(out)


LATEX_PARAM_VALUES: dict[str, str] = {}


def register_latex_parameter_values(genes: list[Any]) -> None:
    """Map fitted parameter labels to numeric values for rendered LaTeX equations."""
    LATEX_PARAM_VALUES.clear()

    for g in genes:
        gene = g.name

        LATEX_PARAM_VALUES[plain_label("A", gene)] = latex_value(g.A)
        LATEX_PARAM_VALUES[plain_label("B", gene)] = latex_value(g.B)
        LATEX_PARAM_VALUES[plain_label("C", gene)] = latex_value(g.C)
        LATEX_PARAM_VALUES[plain_label("D", gene)] = latex_value(g.D)
        LATEX_PARAM_VALUES[plain_label("E", gene)] = latex_value(g.E)
        LATEX_PARAM_VALUES[plain_label("tau", gene)] = latex_value(g.tf_scale)

        for site in g.sites:
            LATEX_PARAM_VALUES[plain_label("S", gene, site)] = latex_value(g.S.get(site, 0.0))
            LATEX_PARAM_VALUES[plain_label("Dp", gene, site)] = latex_value(g.Dp.get(site, 0.0))


def tex_label(prefix: str, *parts: Any, time: Optional[str] = None) -> str:
    raw = plain_label(prefix, *parts)

    # For fitted parameters, render the numeric fitted value directly.
    # This affects A, B, C, D, E, tau, S, and Dp after registration.
    if raw in LATEX_PARAM_VALUES:
        return LATEX_PARAM_VALUES[raw]

    base = latex_math_id(raw)
    if time is None:
        return base
    return base + f"({time})"


def tex_join_signed(terms: list[tuple[str, str]]) -> list[str]:
    """Return aligned RHS lines from signed LaTeX terms."""
    if not terms:
        return ["0"]
    lines: list[str] = []
    current = ""
    for sign, term in terms:
        piece = term if sign == "+" else "- " + term
        if not current:
            current = piece if sign == "+" else "- " + term
        elif len(current) + len(piece) > 115:
            lines.append(current)
            current = ("+ " + term) if sign == "+" else "- " + term
        else:
            current += (" + " + term) if sign == "+" else (" - " + term)
    if current:
        lines.append(current)
    return lines


def append_aligned_equation(out: list[str], lhs: str, terms: list[tuple[str, str]]) -> None:
    rhs_lines = tex_join_signed(terms)
    out.append(lhs + " &= " + rhs_lines[0] + r" \\")
    for extra in rhs_lines[1:]:
        out.append(r"&\quad " + extra + r" \\")


def latex_shared_synthesis(out: list[str], gene: str) -> None:
    A = tex_label("A", gene)
    B = tex_label("B", gene)
    tau = tex_label("tau", gene)
    T = tex_label("T", gene, time="t")
    u = tex_label("u", gene, time="t")
    synth = tex_label("synth", gene, time="t")
    R = tex_label("R", gene, time="t")
    R0 = tex_label("R", gene)
    out.append(r"\begin{align*}")
    out.append(rf"{u} &= \frac{{{T}}}{{1 + \lvert {T} \rvert}} \\")
    out.append(rf"{synth} &= \begin{{cases}}")
    out.append(rf"{A}\left(1 + \dfrac{{{tau}\,{u}}}{{1 + {u} + 10^{{-6}}}}\right), & {u} \ge 0,\\")
    out.append(rf"\dfrac{{{A}}}{{1 + {tau}\,\lvert {u} \rvert}}, & {u} < 0,")
    out.append(r"\end{cases} \\")
    out.append(rf"\frac{{d\,{R0}}}{{dt}} &= {synth} - {B}\,{R} \\")


def latex_distributive_gene(out: list[str], g: Any, *, saturated: bool = False) -> None:
    gene = g.name
    out.append(r"\subsubsection{" + latex_escape(gene) + "}")
    latex_shared_synthesis(out, gene)
    R = tex_label("R", gene, time="t")
    P = tex_label("P", gene, time="t")
    P0 = tex_label("P", gene)
    C = tex_label("C", gene)
    D = tex_label("D", gene)
    E = tex_label("E", gene)
    trans = rf"\frac{{{C}\,{R}}}{{1 + {R}}}" if saturated else rf"{C}\,{R}"
    p_terms: list[tuple[str, str]] = [("+", trans), ("-", rf"{D}\,{P}")]
    for site in g.sites:
        S = tex_label("S", gene, site, time="t")
        X = tex_label("X", gene, site, time="t")
        F = rf"\frac{{{S}\,{P}}}{{1 + {P}}}" if saturated else rf"{S}\,{P}"
        p_terms.append(("-", F))
    for site in g.sites:
        X = tex_label("X", gene, site, time="t")
        p_terms.append(("+", rf"{E}\,{X}"))
    append_aligned_equation(out, rf"\frac{{d\,{P0}}}{{dt}}", p_terms)
    for site in g.sites:
        X = tex_label("X", gene, site, time="t")
        X0 = tex_label("X", gene, site)
        S = tex_label("S", gene, site, time="t")
        Dp = tex_label("Dp", gene, site)
        F = rf"\frac{{{S}\,{P}}}{{1 + {P}}}" if saturated else rf"{S}\,{P}"
        terms = [("+", F), ("-", rf"\left({E} + {Dp} + {D}\right){X}")]
        append_aligned_equation(out, rf"\frac{{d\,{X0}}}{{dt}}", terms)
    out.append(r"\end{align*}")


def latex_successive_gene(out: list[str], g: Any) -> None:
    gene = g.name
    sites = list(g.sites)
    out.append(r"\subsubsection{" + latex_escape(gene) + "}")
    latex_shared_synthesis(out, gene)
    R = tex_label("R", gene, time="t")
    P = tex_label("P", gene, time="t")
    P0 = tex_label("P", gene)
    C = tex_label("C", gene)
    D = tex_label("D", gene)
    E = tex_label("E", gene)
    if not sites:
        append_aligned_equation(out, rf"\frac{{d\,{P0}}}{{dt}}", [("+", rf"{C}\,{R}"), ("-", rf"{D}\,{P}")])
        out.append(r"\end{align*}")
        return
    first = sites[0]
    S1 = tex_label("S", gene, first, time="t")
    X1 = tex_label("X", gene, first, time="t")
    append_aligned_equation(out, rf"\frac{{d\,{P0}}}{{dt}}",
                            [("+", rf"{C}\,{R}"), ("-", rf"{D}\,{P}"), ("-", rf"{S1}\,{P}"), ("+", rf"{E}\,{X1}")])
    for j, site in enumerate(sites):
        X = tex_label("X", gene, site, time="t")
        X0 = tex_label("X", gene, site)
        S = tex_label("S", gene, site, time="t")
        Dp = tex_label("Dp", gene, site)
        upstream = P if j == 0 else tex_label("X", gene, sites[j - 1], time="t")
        terms = [("+", rf"{S}\,{upstream}")]
        if j + 1 < len(sites):
            Xnext = tex_label("X", gene, sites[j + 1], time="t")
            Snext = tex_label("S", gene, sites[j + 1], time="t")
            terms.append(("+", rf"{E}\,{Xnext}"))
            terms.append(("-", rf"\left({Snext} + {E} + {Dp} + {D}\right){X}"))
        else:
            terms.append(("-", rf"\left({E} + {Dp} + {D}\right){X}"))
        append_aligned_equation(out, rf"\frac{{d\,{X0}}}{{dt}}", terms)
    out.append(r"\end{align*}")


def latex_combinatorial_gene(out: list[str], g: Any) -> None:
    gene = g.name
    sites = list(g.sites)
    out.append(r"\subsubsection{" + latex_escape(gene) + "}")
    latex_shared_synthesis(out, gene)
    R = tex_label("R", gene, time="t")
    C = tex_label("C", gene)
    D = tex_label("D", gene)
    E = tex_label("E", gene)
    n_masks = 2 ** len(sites) if sites else 1
    for m in range(n_masks):
        Pm = tex_label("P", gene, f"mask{m}", time="t")
        Pm0 = tex_label("P", gene, f"mask{m}")
        terms: list[tuple[str, str]] = []
        if m == 0:
            terms += [("+", rf"{C}\,{R}"), ("-", rf"{D}\,{Pm}")]
        for j, site in enumerate(sites):
            bit = 1 << j
            S = tex_label("S", gene, site, time="t")
            Dp = tex_label("Dp", gene, site)
            if m & bit:
                Pclear = tex_label("P", gene, f"mask{m ^ bit}", time="t")
                terms += [("+", rf"{S}\,{Pclear}"), ("-", rf"{E}\,{Pm}"), ("-", rf"\left({Dp}+{D}\right){Pm}")]
            else:
                Pset = tex_label("P", gene, f"mask{m | bit}", time="t")
                terms += [("+", rf"{E}\,{Pset}"), ("-", rf"{S}\,{Pm}")]
        append_aligned_equation(out, rf"\frac{{d\,{Pm0}}}{{dt}}", terms)
    out.append(r"\end{align*}")


def latex_network_equations(results_dir: Path, model_id: int) -> str:
    parsed = parse_networkmodel_tables(results_dir)
    genes = build_network_genes(parsed)
    register_latex_parameter_values(genes)
    out: list[str] = []
    topology = network_model_topology(model_id) if "network_model_topology" in globals() else str(model_id)
    out.append(r"\section{Equations}")
    out.append(r"\allowdisplaybreaks")
    out.append(r"Networkmodel topology: \textbf{" + latex_escape(topology) + r"} (MODEL=" + str(model_id) + r").")
    out.append(r"The following equations are rendered directly as LaTeX displays, not as verbatim Markdown.")
    for g in genes:
        if model_id == 1:
            latex_successive_gene(out, g)
        elif model_id == 2:
            latex_combinatorial_gene(out, g)
        elif model_id == 4:
            latex_distributive_gene(out, g, saturated=True)
        else:
            latex_distributive_gene(out, g, saturated=False)
    return "\n".join(out)


def latex_sym(text: Any) -> str:
    return r"\mathrm{" + latex_escape(text) + r"}"


def latex_sub(base: str, sub: Any) -> str:
    return latex_sym(base) + r"_{" + latex_sym(sub) + r"}"


def protwise_param_names(params: dict[str, float], prefix: str) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        rest = str(name)[len(prefix):]
        return (int(rest), str(name)) if rest.isdigit() else (10 ** 9, str(name))

    return sorted(
        [str(k) for k in params if str(k).startswith(prefix) and str(k)[len(prefix):].isdigit()],
        key=key,
    )


def protwise_value(params: dict[str, float], name: str, default: float = 0.0) -> str:
    return latex_value(params.get(name, default))


def latex_protwise_equations(results_dir: Path) -> str:
    parsed = parse_protwise_parameters(results_dir)
    out: list[str] = []

    out.append(r"\section{Equations}")
    out.append(r"\allowdisplaybreaks")
    out.append(
        "Protwise equations are rendered separately for each protein using the fitted values "
        "from its final parameter row. For distributive models, every site state is fed from "
        "the unphosphorylated protein pool. For successive models, site states form an ordered chain."
    )

    for entry in parsed["gene_params"]:
        gene = entry["gene"]
        params = entry["params"]
        model_kind = protwise_model_kind(entry)

        A = protwise_value(params, "A")
        B = protwise_value(params, "B")
        C = protwise_value(params, "C")
        D = protwise_value(params, "D")

        s_names = protwise_param_names(params, "S")
        n_sites = len(s_names)

        R = latex_sub("R", gene)
        Rt = R + "(t)"
        P = latex_sub("P", gene)
        Pt = P + "(t)"

        out.append(r"\subsection{" + latex_escape(gene) + r"}")
        out.append(r"Detected Protwise topology: \texttt{" + latex_escape(model_kind) + r"}.")
        out.append(r"\begin{align*}")
        out.append(rf"\frac{{d\,{R}}}{{dt}} &= {A} - {B}\,{Rt} \\")

        if model_kind == "succmod":
            if n_sites == 0:
                out.append(rf"\frac{{d\,{P}}}{{dt}} &= {C}\,{Rt} - {D}\,{Pt} \\")
            else:
                X1 = latex_sub("X1", gene)
                S1 = protwise_value(params, s_names[0])
                out.append(
                    rf"\frac{{d\,{P}}}{{dt}} &= {C}\,{Rt} - {D}\,{Pt} "
                    rf"- {S1}\,{Pt} + {X1}(t) \\"
                )

                for i, s_name in enumerate(s_names, start=1):
                    Xi = latex_sub(f"X{i}", gene)
                    Xit = Xi + "(t)"
                    Si = protwise_value(params, s_name)
                    Di = protwise_value(params, f"D{i}")

                    upstream = Pt if i == 1 else latex_sub(f"X{i - 1}", gene) + "(t)"

                    if i < n_sites:
                        Xnext = latex_sub(f"X{i + 1}", gene) + "(t)"
                        Snext = protwise_value(params, f"S{i + 1}")
                        out.append(
                            rf"\frac{{d\,{Xi}}}{{dt}} &= {Si}\,{upstream} + {Xnext} "
                            rf"- \left({Snext} + 1 + {Di} + {D}\right){Xit} \\"
                        )
                    else:
                        out.append(
                            rf"\frac{{d\,{Xi}}}{{dt}} &= {Si}\,{upstream} "
                            rf"- \left(1 + {Di} + {D}\right){Xit} \\"
                        )

        elif model_kind == "randmod":
            out.append(
                rf"\frac{{d\,{P}}}{{dt}} &= {C}\,{Rt} - {D}\,{Pt}"
                r"\quad\text{ plus random/combinatorial phosphorylation-state transitions.}\\"
            )
            out.append(
                r"\text{Random-model state equations are not expanded here because the explicit state-combination schema is not stored in the fitted parameter table.}\\"
            )

        else:
            p_loss_terms = []
            p_gain_terms = []

            for i, s_name in enumerate(s_names, start=1):
                Si = protwise_value(params, s_name)
                Xi = latex_sub(f"X{i}", gene)
                p_loss_terms.append(rf"{Si}\,{Pt}")
                p_gain_terms.append(rf"{Xi}(t)")

            rhs_p = rf"{C}\,{Rt} - {D}\,{Pt}"
            for term in p_loss_terms:
                rhs_p += rf" - {term}"
            for term in p_gain_terms:
                rhs_p += rf" + {term}"

            out.append(rf"\frac{{d\,{P}}}{{dt}} &= {rhs_p} \\")

            for i, s_name in enumerate(s_names, start=1):
                Xi = latex_sub(f"X{i}", gene)
                Xit = Xi + "(t)"
                Si = protwise_value(params, s_name)
                Di = protwise_value(params, f"D{i}")
                out.append(
                    rf"\frac{{d\,{Xi}}}{{dt}} &= {Si}\,{Pt} "
                    rf"- \left(1 + {Di} + {D}\right){Xit} \\"
                )

        out.append(r"\end{align*}")

    return "\n".join(out)


def latex_kinopt_equations(results_dir: Path) -> str:
    wb = first_existing(results_dir, ("kinopt_results.xlsx",))
    if wb is None:
        raise FileNotFoundError("Could not find kinopt_results.xlsx")

    alpha, beta = parse_kinopt_xlsx(wb)
    out: list[str] = []

    out.append(r"\section{Equations}")
    out.append(
        "KinOpt is documented as a fitted coefficient model. "
        r"Kinase-site activities are first aggregated with fitted \(\beta\) coefficients; "
        r"target phosphorylation is then reconstructed with fitted \(\alpha\) coefficients."
    )

    out.append(r"\subsection{Kinase activity definitions}")
    for kinase, sub in beta.groupby("Kinase", sort=True):
        terms = []
        for _, row in sub.iterrows():
            val = latex_value(row["Beta"])
            psite = latex_escape(row["Psite"])
            terms.append(rf"{val}\,x_{{{latex_sym(kinase)},{latex_sym(psite)}}}(t)")
        rhs = " + ".join(terms) if terms else "0"
        out.append(r"\begin{align*}")
        out.append(rf"K_{{{latex_sym(kinase)}}}(t) &= {rhs}")
        out.append(r"\end{align*}")

    out.append(r"\subsection{Target phosphorylation equations}")
    for (gene, psite), sub in alpha.groupby(["Gene", "Psite"], sort=True):
        terms = []
        for _, row in sub.iterrows():
            val = latex_value(row["Alpha"])
            kinase = row["Kinase"]
            terms.append(rf"{val}\,K_{{{latex_sym(kinase)}}}(t)")
        rhs = " + ".join(terms) if terms else "0"
        out.append(r"\begin{align*}")
        out.append(
            rf"\widehat{{X}}_{{{latex_sym(gene)},{latex_sym(psite)}}}(t) &= {rhs}"
        )
        out.append(r"\end{align*}")

    return "\n".join(out)


def latex_tfopt_equations(results_dir: Path) -> str:
    wb = first_existing(results_dir, ("tfopt_results.xlsx",))
    if wb is None:
        raise FileNotFoundError("Could not find tfopt_results.xlsx")

    alpha, beta = parse_tfopt_xlsx(wb)
    out: list[str] = []

    out.append(r"\section{Equations}")
    out.append(
        "TFOpt is documented as a fitted coefficient model. "
        r"TF phosphosite activities are first aggregated with fitted \(\beta\) coefficients; "
        r"mRNA regulation is then reconstructed with fitted \(\alpha\) coefficients."
    )

    out.append(r"\subsection{TF activity definitions}")
    for tf, sub in beta.groupby("TF", sort=True):
        terms = []
        for _, row in sub.iterrows():
            val = latex_value(row["Value"])
            psite = latex_escape(row["PSite"])
            terms.append(rf"{val}\,x_{{{latex_sym(tf)},{latex_sym(psite)}}}(t)")
        rhs = " + ".join(terms) if terms else "0"
        out.append(r"\begin{align*}")
        out.append(rf"T_{{{latex_sym(tf)}}}(t) &= {rhs}")
        out.append(r"\end{align*}")

    out.append(r"\subsection{mRNA regulation equations}")
    for mrna, sub in alpha.groupby("mRNA", sort=True):
        terms = []
        for _, row in sub.iterrows():
            val = latex_value(row["Value"])
            tf = row["TF"]
            terms.append(rf"{val}\,T_{{{latex_sym(tf)}}}(t)")
        rhs = " + ".join(terms) if terms else "0"
        out.append(r"\begin{align*}")
        out.append(rf"\widehat{{R}}_{{{latex_sym(mrna)}}}(t) &= {rhs}")
        out.append(r"\end{align*}")

    return "\n".join(out)


def latex_metadata_section(results_dir: Path, family: str, metadata: dict[str, Any]) -> str:
    lines = [
        r"\section{Run metadata}",
        r"\begin{description}",
        r"\item[Result directory] \texttt{" + latex_escape(results_dir) + r"}",
        r"\item[Model family] \texttt{" + latex_escape(family) + r"}",
    ]
    if "networkmodel_model_id" in metadata:
        lines.append(r"\item[Network MODEL] \texttt{" + str(metadata.get("networkmodel_model_id")) + r"}")
    lines.append(r"\end{description}")
    return "\n".join(lines)


def build_latex(results_dir: Path, family: str, metadata: dict[str, Any], tables: list[tuple[str, pd.DataFrame]],
                equations: str) -> str:
    """Build a real LaTeX document: full landscape tables plus rendered equations."""
    parts: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=0.75in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage{booktabs,longtable,array}",
        r"\usepackage{pdflscape}",
        r"\usepackage{hyperref}",
        r"\setcounter{secnumdepth}{3}",
        r"\setcounter{tocdepth}{2}",
        r"\title{PhosKinTime Model Documentation}",
        rf"\author{{{latex_escape(family)}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        latex_metadata_section(results_dir, family, metadata),
        r"\section{Fitted parameter tables}",
        r"The following tables are printed in full. They are rotated to landscape using \texttt{pdflscape} and split across pages with \texttt{longtable} when necessary.",
    ]

    for title, df in tables:
        parts.append(latex_longtable(title, df))

    if family == "networkmodel":
        parts.append(
            latex_network_equations(
                results_dir,
                int(metadata.get("networkmodel_model_id", infer_network_model_id(results_dir))),
            )
        )
    elif family == "protwise":
        parts.append(latex_protwise_equations(results_dir))
    elif family == "kinopt":
        parts.append(latex_kinopt_equations(results_dir))
    elif family == "tfopt":
        parts.append(latex_tfopt_equations(results_dir))
    else:
        parts.append(r"\section{Equations}")
        parts.append(latex_escape(equations))

    parts.extend([
        r"\section{Internal metadata}",
        r"\begin{verbatim}",
        json.dumps(metadata, indent=2),
        r"\end{verbatim}",
        r"\end{document}",
        "",
    ])

    return "\n".join(parts)


def compile_pdf(tex_path: Path, output_dir: Path) -> Optional[Path]:
    engine = shutil.which("tectonic")
    if engine:
        cmd = [engine, str(tex_path), "--outdir", str(output_dir)]
    else:
        engine = shutil.which("pdflatex")
        if not engine:
            return None
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(output_dir),
               str(tex_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    pdf = output_dir / "model_documentation.pdf"
    return pdf if pdf.is_file() else None


def generate_documentation(results_dir: Path, output_dir: Path, family: str, verbose: bool = False) -> dict[
    str, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if family == "networkmodel":
        metadata, tables, equations = network_metadata_and_tables(results_dir)
    elif family == "protwise":
        metadata, tables, equations = protwise_metadata_and_tables(results_dir)
    elif family in {"kinopt", "tfopt"}:
        metadata, tables, equations = kin_tf_metadata_and_tables(results_dir, family)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    metadata["result_dir"] = str(results_dir)
    metadata["model_family"] = family
    md = build_markdown(results_dir, family, metadata, tables, equations)
    md_path = output_dir / "model_documentation.md"
    tex_path = output_dir / "model_documentation.tex"
    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(build_latex(results_dir, family, metadata, tables, equations), encoding="utf-8")
    pdf_path = compile_pdf(tex_path, output_dir)
    if pdf_path is None and verbose:
        print("Warning: LaTeX engine not available or PDF compilation failed. Wrote Markdown and LaTeX only.")
    return {"md": md_path, "tex": tex_path, "pdf": pdf_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PhosKinTime model documentation.")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-family", choices=["networkmodel", "protwise", "kinopt", "tfopt"], default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)
    family, family_source = resolve_family(results_dir, args.model_family)
    if args.verbose:
        print(f"Model family: {family} ({family_source})")
        print(f"Results dir: {results_dir}")
        print(f"Output dir: {output_dir}")

    outputs = generate_documentation(results_dir, output_dir, family, verbose=args.verbose)

    print("\nFinal summary")
    print("Implementation grounding:")
    if family == "networkmodel":
        print("- networkmodel equations grounded in networkmodel/backend.py::make_networkmodel_rhs")
        print("- networkmodel objective/observables grounded in multimodal_loss_from_trajectory")
    elif family == "protwise":
        print(
            "- protwise equations grounded in actual ODE/Diffrax implementations and existing LaTeX files when present")
    else:
        print("- KinOpt/TFOpt schemas grounded in actual workbook export sheets")
    print("\nOutputs:")
    for k, p in outputs.items():
        print(f"- {k}: {p if p else 'not generated'}")
    print("- Markdown, LaTeX, and PDF documentation written where possible")


if __name__ == "__main__":
    main()
