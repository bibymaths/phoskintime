#!/usr/bin/env python3
"""Export a fitted PhosKinTime DISTRIBUTIVE network as reaction-based SBML L3V2.

No fitting or ODE integration is performed. Model topology, signed TF weights,
kinase-input trajectories, fitted physical parameters and initial states are
recovered from the selected fitted run's own saved networkmodel exports.
The generated reaction-form SBML is compared with the JAX fitting RHS before
writing; failures never silently truncate parameter vectors or invent edges.

This intentionally supports only networkmodel MODEL=0.  Other model families
and kinetic topologies must have separately tested reaction definitions.

Usage (from repository root):
    python scripts/export/export_model_to_sbml.py \\
      --results results_network_distributive \\
      --out results_network_distributive/sbml_reactions \\
      --conf config.toml
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

try:
    import libsbml
except ImportError:  # Plan-generation and pure-Python tests still work.
    libsbml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "networkmodel").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ReactionDef:
    id: str
    name: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    modifiers: tuple[str, ...]
    formula: str
    reversible: bool = False


@dataclass
class ExportPlan:
    genes: list[str]
    sites: dict[str, list[str]]
    kinases: list[str]
    species: dict[str, float | None] = field(default_factory=dict)
    parameters: dict[str, float | None] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    reactions: list[ReactionDef] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_kinase_grid: list[float] = field(default_factory=list)
    raw_kinase_matrix: list[list[float]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def sid(prefix: str, *parts: object) -> str:
    """Unique-by-construction SIds are checked explicitly when added."""
    tokens = []
    for part in (prefix,) + parts:
        s = re.sub(r"[^A-Za-z0-9_]", "_", str(part))
        s = re.sub(r"_+", "_", s).strip("_")
        if not s:
            raise ValueError(f"Empty SBML identifier token: {part!r}")
        tokens.append(s)
    result = "_".join(tokens)
    if not re.match(r"^[A-Za-z_]", result):
        result = "id_" + result
    return result


def num(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in SBML: {value}")
    return format(value, ".17g")


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _require_finite_nonnegative(value: Any, label: str) -> float:
    f = float(value)
    if not np.isfinite(f) or f < 0:
        raise ValueError(f"{label} must be a nonnegative finite fitted physical parameter, got {value!r}")
    return f


def read_fitted_params(results: Path, idx) -> dict[str, np.ndarray | float]:
    path = results / "fitted_params_picked.json"
    if not path.is_file():
        raise FileNotFoundError(f"Need chosen fitted physical parameters: {path}")
    data = json.loads(path.read_text())
    sizes = {"A_i": idx.N, "B_i": idx.N, "C_i": idx.N, "D_i": idx.N,
             "E_i": idx.N, "Dp_i": idx.total_sites, "c_k": len(idx.kinases)}
    fitted = {}
    for name, length in sizes.items():
        if name not in data:
            raise KeyError(f"Fitted parameter {name!r} missing from {path}")
        values = np.asarray(data[name], dtype=float).ravel()
        if values.size != length:
            raise ValueError(f"{name}: expected {length} entries, got {values.size}")
        for v in values:
            _require_finite_nonnegative(v, name)
        fitted[name] = values
    if "tf_scale" not in data:
        raise KeyError("Missing tf_scale")
    fitted["tf_scale"] = _require_finite_nonnegative(data["tf_scale"], "tf_scale")
    return fitted


def _edge_map(df: pd.DataFrame, keys: tuple[str, ...], value_col: str = "Weight") -> dict[tuple, float]:
    missing = set(keys) | {value_col}
    missing -= set(df.columns)
    if missing:
        raise ValueError(f"Missing exported network columns: {sorted(missing)}")
    result = defaultdict(float)
    for row in df.to_dict("records"):
        value = float(row[value_col])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite exported edge weight: {row}")
        result[tuple(str(row[k]) for k in keys)] += value
    return {k: v for k, v in result.items() if abs(v) > 1e-15}


def _validate_edges(actual: dict[tuple, float], exported: dict[tuple, float], label: str) -> None:
    allkeys = set(actual) | set(exported)
    diffs = [(k, actual.get(k, 0), exported.get(k, 0))
             for k in sorted(allkeys)
             if not np.isclose(actual.get(k, 0), exported.get(k, 0), rtol=1e-6, atol=1e-9)]
    if diffs:
        raise ValueError(f"{label} differs from the saved network. First differences: {diffs[:5]}. "
                         "The original input files or network preprocessing may have changed; refuse to mix fits.")


def check_rebuilt_network(results: Path, sys_obj) -> dict[str, int]:
    """Cross-check against the existing PhosKinTime export schema."""
    idx = sys_obj.idx
    W = sys_obj.W_global.tocoo()
    actual_w = defaultdict(float)
    site_keys = [(g, site) for i, g in enumerate(idx.proteins) for site in idx.sites[i]]
    for r, c, v in zip(W.row, W.col, W.data):
        if not np.isfinite(v):
            raise ValueError("Invalid W element")
        actual_w[(f"{site_keys[int(r)][0]}_{site_keys[int(r)][1]}", str(idx.kinases[int(c)]))] += float(v)
    w_df = _read(results / "network_W_global.csv")
    _validate_edges(actual_w, _edge_map(w_df, ("Site", "Kinase")), "Kinase-to-site matrix")

    TF = sys_obj.tf_mat.tocoo()
    actual_tf = defaultdict(float)
    for r, c, v in zip(TF.row, TF.col, TF.data):
        actual_tf[(str(idx.proteins[int(r)]), str(idx.proteins[int(c)]))] += float(v)
    t_df = _read(results / "network_tf_mat.csv")
    _validate_edges(actual_tf, _edge_map(t_df, ("Target", "Source_TF")), "TF-to-gene matrix")

    genes_df = _read(results / "model_parameters_genes.csv")
    if "Protein_Gene" not in genes_df:
        raise ValueError("Missing Protein_Gene column in fitted gene table")
    if set(map(str, genes_df["Protein_Gene"])) != set(map(str, idx.proteins)):
        raise ValueError("Gene set in saved fitted results differs from rebuilt network")
    return {"kinase_site_edges": len(actual_w), "tf_gene_edges": len(actual_tf)}


def check_fitted_parameter_tables(results: Path, idx, par: dict) -> None:
    gen = _read(results / "model_parameters_genes.csv").set_index("Protein_Gene")
    mapping = {"A_i": "Synthesis_A", "B_i": "mRNA_Degradation_B", "C_i": "Translation_C",
               "D_i": "Protein_Degradation_D", "E_i": "De-Phosphorylation_E"}
    for i, gene in enumerate(idx.proteins):
        for key, col in mapping.items():
            if not np.isclose(float(gen.loc[gene, col]), par[key][i], rtol=1e-6, atol=1e-9):
                raise ValueError(f"Fitted {key}[{gene}] differs between JSON and gene table")
        if not np.isclose(float(gen.loc[gene, "Global_TF_Scale"]), par["tf_scale"], rtol=1e-6):
            raise ValueError("Fitted TF scale mismatch")
    kin = _read(results / "model_parameters_kinases.csv").set_index("Kinase")
    for k, v in zip(idx.kinases, par["c_k"]):
        if not np.isclose(float(kin.loc[k, "Activity_Scale_ck"]), v, rtol=1e-6, atol=1e-9):
            raise ValueError(f"Fitted c_k[{k}] mismatch")
    psite_file = results / "model_parameters_genes_psites.csv"
    if psite_file.exists():
        site_df = _read(psite_file)
        if {"Protein_Gene", "psite", "Phospho_Degradation_Dp"} <= set(site_df.columns):
            site_dps = {(str(r["Protein_Gene"]), str(r["psite"])): float(r["Phospho_Degradation_Dp"])
                        for _, r in site_df.dropna(subset=["psite"]).iterrows()}
            for i, gene in enumerate(idx.proteins):
                for j, site in enumerate(idx.sites[i]):
                    if (gene, site) in site_dps and not np.isclose(
                        site_dps[(gene, site)], par["Dp_i"][int(idx.offset_s[i]) + j], rtol=1e-6, atol=1e-9
                    ):
                        raise ValueError(f"Fitted phosphosite Dp mismatch: {gene} {site}")


def check_phospho_drive_samples(results: Path, sys_obj, par: dict) -> int:
    path = results / "S_rates_picked.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot verify reconstructed time-varying drives without {path}")
    df = pd.read_csv(path)
    if not {"protein", "psite", "time", "S"} <= set(df.columns):
        raise ValueError("S_rates_picked.csv does not have the PhosKinTime long schema")
    idx = sys_obj.idx
    row_map = {(gene, site): int(idx.offset_s[i]) + j
               for i, gene in enumerate(idx.proteins) for j, site in enumerate(idx.sites[i])}
    for r in df.itertuples(index=False):
        key = (str(r.protein), str(r.psite))
        if key not in row_map:
            raise ValueError(f"Saved phosphorylation-drive site absent from model: {key}")
        test = np.asarray(sys_obj.W_global @ (np.asarray(sys_obj.kin.eval(float(r.time))) * par["c_k"])).ravel()
        if not np.isclose(float(r.S), float(test[row_map[key]]), rtol=1e-5, atol=1e-8):
            raise ValueError(f"Drive mismatch for {key} at t={r.time}: saved={r.S} rebuilt={test[row_map[key]]}")
    return len(df)


def check_initial_conditions_table(results: Path, sys_obj):
    """Validate the ACTUAL long-schema initial_conditions_y0.csv, not a guessed vector."""
    file = results / "initial_conditions_y0.csv"
    df = _read(file)
    required = {"entity", "kind", "substate", "value"}
    if not required <= set(df.columns):
        raise ValueError(f"Initial condition table must contain {required}, got {set(df.columns)}")
    actual = {(str(row.entity), str(row.kind), str(row.substate)): float(row.value)
              for row in df.itertuples(index=False)}
    idx = sys_obj.idx
    y = np.asarray(sys_obj.y0(), dtype=float)
    checked = 0
    for i, gene in enumerate(idx.proteins):
        off = int(idx.offset_y[i])
        items = [("mRNA", "R", y[off]), ("protein_state", "P0", y[off+1])]
        items.extend(("phospho_state", str(site), y[off+2+j]) for j, site in enumerate(idx.sites[i]))
        for kind, substate, value in items:
            key = (gene, kind, substate)
            if key not in actual:
                raise ValueError(f"Initial state missing from saved table: {key}")
            if not np.isclose(value, actual[key], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Rebuilt initial state differs from saved: {key}: {value} vs {actual[key]}")
            checked += 1
    return checked


def piecewise_stepwise(times: np.ndarray, levels: np.ndarray) -> str:
    """Right-continuous sample-and-hold; mirrors KinaseInput.eval/ backend._stepwise."""
    times = np.asarray(times, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if len(times) != len(levels) or len(times) < 1:
        raise ValueError("Kinase times and levels must be the same nonzero length")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(levels)) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid kinase input grid or values")
    if len(levels) == 1:
        return num(levels[0])
    # e.g. piecewise(v0, time < t1, v1, time < t2, v2) (right-continuous).
    terms = []
    for i in range(len(times)-1):
        terms += [num(levels[i]), f"lt(time, {num(times[i+1])})"]
    terms.append(num(levels[-1]))
    return f"piecewise({', '.join(terms)})"


def _species_ids(genes: list[str], sites: dict[str, list[str]], kinases: list[str]):
    all_ids = [sid("R", g) for g in genes] + [sid("P", g) for g in genes]
    all_ids += [sid("X", g, site) for g in genes for site in sites[g]]
    all_ids += [sid("K", k) for k in kinases]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("SBML SId collision after sanitizing gene/site names")


def make_plan(sys_obj, fitted: dict[str, Any]) -> ExportPlan:
    """Build reaction and assignment definitions directly from System topology."""
    idx = sys_obj.idx
    genes = list(map(str, idx.proteins))
    kinases = list(map(str, idx.kinases))
    sites = {str(g): list(map(str, idx.sites[i])) for i, g in enumerate(genes)}
    _species_ids(genes, sites, kinases)
    plan = ExportPlan(genes, sites, kinases)
    grid = np.asarray(sys_obj.kin.grid, dtype=float)
    Kmat = np.asarray(sys_obj.kin.Kmat, dtype=float)
    if Kmat.shape != (len(kinases), len(grid)):
        raise ValueError("Rebuilt KinaseInput time grid does not match data matrix")
    plan.raw_kinase_grid = grid.tolist()
    plan.raw_kinase_matrix = Kmat.tolist()
    all_drives = np.asarray(sys_obj.W_global @ (Kmat * np.asarray(fitted["c_k"])[:, None]))
    if not np.all(np.isfinite(all_drives)):
        raise ValueError("Nonfinite fitted site phosphorylation drives")
    if np.any(all_drives < -1e-10):
        plan.notes.append("Some fitted phosphorylation drives are negative; these imply signed ")
        plan.notes.append("effective reversible fluxes, not conventional nonnegative enzyme rates.")
    y0 = np.asarray(sys_obj.y0(), dtype=float)
    if y0.size != idx.state_dim or np.any(~np.isfinite(y0)):
        raise ValueError("Initial-condition state vector invalid")
    if np.any(y0 < 0):
        raise ValueError("Negative initial state would invalidate the reaction interpretation")

    def add_species(key: str, initial: float | None):
        if key in plan.species or key in plan.parameters:
            raise ValueError(f"Duplicate SBML identifier: {key}")
        plan.species[key] = initial

    def add_param(key: str, value: float | None):
        if key in plan.parameters or key in plan.species:
            raise ValueError(f"Duplicate SBML identifier: {key}")
        plan.parameters[key] = value

    def add_assignment(key: str, expr: str):
        if key not in plan.parameters and key not in plan.species:
            raise ValueError(f"Undefined assignment target: {key}")
        if key in plan.assignments:
            raise ValueError(f"Duplicate assignment target: {key}")
        plan.assignments[key] = expr

    def add_reaction(kind: str, label: str, reactants: list[str], products: list[str],
                     modifiers: list[str], formula: str, reversible: bool = False):
        rid = sid("rx", kind, label)
        if any(r.id == rid for r in plan.reactions):
            raise ValueError(f"Duplicate SBML reaction: {rid}")
        all_participants = reactants + products + modifiers
        unknown = set(all_participants) - set(plan.species)
        if unknown:
            raise ValueError(f"Reaction {rid} references unknown species {unknown}")
        plan.reactions.append(ReactionDef(rid, f"{kind} {label}", tuple(reactants),
                                          tuple(products), tuple(dict.fromkeys(modifiers)), formula, reversible))

    # Actual observed kinase-input time series are boundary species (not inferred states).
    for k, kinase in enumerate(kinases):
        kid = sid("K", kinase)
        ckid = sid("ck", kinase)
        add_species(kid, None)
        add_param(ckid, float(fitted["c_k"][k]))
        add_assignment(kid, f"{ckid} * ({piecewise_stepwise(grid, Kmat[k])})")

    # Create all dynamic species/parameters first to make cross-references explicit.
    for i, gene in enumerate(genes):
        off = int(idx.offset_y[i])
        add_species(sid("R", gene), float(y0[off]))
        add_species(sid("P", gene), float(y0[off+1]))
        for j, site in enumerate(sites[gene]):
            add_species(sid("X", gene, site), float(y0[off+2+j]))
        for pname, short in (("A_i", "A"), ("B_i", "B"), ("C_i", "C"),
                             ("D_i", "D"), ("E_i", "E")):
            add_param(sid(short, gene), float(fitted[pname][i]))
        for j, site in enumerate(sites[gene]):
            add_param(sid("Dp", gene, site), float(fitted["Dp_i"][int(idx.offset_s[i])+j]))
        for derived in ("Ptotal", "TFinput", "q", "synth"):
            add_param(sid(derived, gene), None)
    add_param("tf_scale", float(fitted["tf_scale"]))

    # Explicit phosphorylating kinases are derived from W and the fitted c_k.
    W = sys_obj.W_global.tocsr()
    kinase_by_index = {i: k for i, k in enumerate(kinases)}
    site_modifiers: dict[tuple[str, str], list[str]] = {}
    for i, gene in enumerate(genes):
        for j, site in enumerate(sites[gene]):
            row = int(idx.offset_s[i]) + j
            w_row = W.getrow(row)
            terms = []
            mods = []
            for k_idx, w in zip(w_row.indices, w_row.data):
                if not np.isfinite(w):
                    raise ValueError("Non-finite W entry")
                kid = sid("K", kinase_by_index[int(k_idx)])
                terms.append(f"({num(w)}) * {kid}")
                mods.append(kid)
            s_id = sid("S", gene, site)
            add_param(s_id, None)
            add_assignment(s_id, " + ".join(terms) if terms else "0")
            site_modifiers[(gene, site)] = mods

    # TF input: signed degree-normalized TF row @ P_vec, exactly as backend.py.
    TF = sys_obj.tf_mat.tocsr()
    tf_deg = np.asarray(sys_obj.tf_deg, dtype=float).ravel()
    if tf_deg.shape != (len(genes),):
        raise ValueError("TF degree length mismatch")
    if np.any(~np.isfinite(tf_deg)) or np.any(tf_deg <= 0):
        raise ValueError("Invalid TF degree normalization")
    kinase_set = set(kinases)
    proxies = dict(getattr(idx, "proxy_map", {}))
    for i, gene in enumerate(genes):
        source_terms = []
        regulators = []
        for pidx, w in zip(TF.getrow(i).indices, TF.getrow(i).data):
            source = genes[int(pidx)]
            kinase_proxy = source if source in kinase_set else proxies.get(source)
            if kinase_proxy is not None:
                if kinase_proxy not in kinase_set:
                    raise ValueError(f"Invalid kinase proxy {source} => {kinase_proxy}")
                driver_expr = sid("K", kinase_proxy)
                regulators.append(driver_expr)
            else:
                driver_expr = sid("Ptotal", source)
                regulators.append(sid("P", source))
                regulators.extend(sid("X", source, site) for site in sites[source])
            source_terms.append(f"({num(w)}) * {driver_expr}")
        tfid = sid("TFinput", gene)
        qid = sid("q", gene)
        add_assignment(tfid, f"({' + '.join(source_terms) if source_terms else '0'}) / ({num(tf_deg[i])})")
        add_assignment(qid, f"{tfid} / (1 + abs({tfid}))")
        add_assignment(sid("Ptotal", gene), " + ".join(
            [sid("P", gene)] + [sid("X", gene, s) for s in sites[gene]]))
        # Directly mirrors networkmodel/backend.py::_synth and networkmodel/models.py.
        synth = (f"piecewise({sid('A',gene)} * (1 + tf_scale * {qid} / "
                 f"(1 + {qid} + 0.000001)), geq({qid}, 0), "
                 f"{sid('A',gene)} / (1 + tf_scale * abs({qid})))")
        add_assignment(sid("synth", gene), synth)

        R, P = sid("R", gene), sid("P", gene)
        B, C, D, E = (sid(t, gene) for t in ("B", "C", "D", "E"))
        add_reaction("transcription", gene, [], [R], regulators, sid("synth", gene))
        add_reaction("mrna_decay", gene, [R], [], [], f"{B} * {R}")
        add_reaction("translation", gene, [], [P], [R], f"{C} * {R}")
        add_reaction("protein_decay", gene, [P], [], [], f"{D} * {P}")

        for site in sites[gene]:
            X = sid("X", gene, site)
            S = sid("S", gene, site)
            Dp = sid("Dp", gene, site)
            site_index = int(idx.offset_s[i]) + sites[gene].index(site)
            signed_drive = bool(np.any(all_drives[site_index] < -1e-10))
            add_reaction("phosphorylation", f"{gene}_{site}", [P], [X],
                         site_modifiers[(gene, site)], f"{S} * {P}", reversible=signed_drive)
            add_reaction("dephosphorylation", f"{gene}_{site}", [X], [P], [], f"{E} * {X}")
            add_reaction("phospho_decay", f"{gene}_{site}", [X], [], [], f"({D} + {Dp}) * {X}")

    plan.counts = {"genes": len(genes), "kinases": len(kinases),
                   "sites": sum(map(len, sites.values())), "species": len(plan.species),
                   "reactions": len(plan.reactions), "assignments": len(plan.assignments),
                   "reaction_modifiers": sum(len(r.modifiers) for r in plan.reactions)}
    # Basic structural invariants: network requires nonzero reaction and modifier counts.
    if not plan.reactions or not plan.counts["reaction_modifiers"]:
        raise ValueError("Empty reaction/modifier graph; cannot export for cy3sbml")
    plan.notes += ["Exported from fitted PhosKinTime networkmodel (MODEL=0 distributive).",
                  "Every reaction and modifier is explicit for reaction graph visualization.",
                  "Fitted physical parameters are read from fitted_params_picked.json.",
                  "Kinase inputs are observed piecewise-constant time series, multiplied by fitted gains c_k.",
                  "TF input uses TF @ P_vec with degree normalization and kinase/proxy driver mapping.",
                  "States and time are dimensionless in this SBML representation (fit uses normalized fold changes).",
                  "Do not reinterpret experimental kinase abundance proxies as measured catalytic activities."]
    return plan


def independent_distributive_rhs(plan: ExportPlan, sys_obj, fitted: dict, t: float, y: np.ndarray) -> np.ndarray:
    """Independent numerical implementation to compare with networkmodel.System.rhs."""
    idx = sys_obj.idx
    y = np.asarray(y, dtype=float)
    dydt = np.zeros_like(y)
    K = np.asarray(sys_obj.kin.eval(float(t))) * fitted["c_k"]
    rates = np.asarray(sys_obj.W_global @ K).ravel()
    raw_tf = []
    for i, gene in enumerate(plan.genes):
        vals = []
        for j in range(len(plan.sites[gene])):
            vals.append(y[int(idx.offset_y[i])+2+j])
        if gene in plan.kinases:
            driver = K[plan.kinases.index(gene)]
        elif gene in getattr(idx, "proxy_map", {}):
            driver = K[plan.kinases.index(idx.proxy_map[gene])]
        else:
            driver = y[int(idx.offset_y[i])+1] + sum(vals)
        raw_tf.append(driver)
    tf = np.asarray(sys_obj.tf_mat @ np.asarray(raw_tf)).ravel() / np.asarray(sys_obj.tf_deg)
    for i, gene in enumerate(plan.genes):
        off = int(idx.offset_y[i])
        s_off = int(idx.offset_s[i])
        R, P = y[off], y[off+1]
        q = tf[i] / (1 + abs(tf[i]))
        A = fitted["A_i"][i]
        scal = fitted["tf_scale"]
        synth = A * (1 + scal*q/(1+q+1e-6)) if q >= 0 else A/(1+scal*abs(q))
        dydt[off] = synth - fitted["B_i"][i]*R
        dydt[off+1] = fitted["C_i"][i]*R - fitted["D_i"][i]*P
        for j, site in enumerate(plan.sites[gene]):
            ss = s_off + j
            X = y[off+2+j]
            fw = rates[ss]*P
            bw = fitted["E_i"][i]*X
            loss = (fitted["D_i"][i] + fitted["Dp_i"][ss])*X
            dydt[off+1] += -fw + bw
            dydt[off+2+j] = fw - bw - loss
    return dydt




def evaluate_reaction_plan_rhs(plan: ExportPlan, sys_obj, t: float, y: np.ndarray) -> np.ndarray:
    """Evaluate the EXPORTED assignment/kinetic-law expressions and stoichiometry.

    A safe AST subset is used solely for validation; arbitrary Python code and
    unknown functions are rejected.  This checks what the SBML contains rather
    than retesting a second handwritten model RHS.
    """
    idx = sys_obj.idx
    y = np.asarray(y, dtype=float)
    env: dict[str, float] = {k: float(v) for k, v in plan.parameters.items() if v is not None}
    for i, gene in enumerate(plan.genes):
        off = int(idx.offset_y[i])
        env[sid("R", gene)] = float(y[off])
        env[sid("P", gene)] = float(y[off+1])
        for j, site in enumerate(plan.sites[gene]):
            env[sid("X", gene, site)] = float(y[off+2+j])
    env["time"] = float(t)
    visiting: set[str] = set()

    def resolve(name: str) -> float:
        if name in env:
            return env[name]
        if name not in plan.assignments:
            raise KeyError(f"Undefined symbol in generated SBML expression: {name}")
        if name in visiting:
            raise ValueError(f"Cyclic assignment in generated SBML: {name}")
        visiting.add(name)
        val = expression(plan.assignments[name])
        visiting.remove(name)
        if not np.isfinite(val):
            raise ValueError(f"Nonfinite assignment evaluation: {name}")
        env[name] = float(val)
        return env[name]

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return resolve(node.id)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fname = node.func.id
            if fname == "piecewise":
                if len(node.args) < 3 or len(node.args) % 2 != 1:
                    raise ValueError("Invalid SBML piecewise expression")
                for j in range(0, len(node.args)-1, 2):
                    if visit(node.args[j+1]):
                        return visit(node.args[j])
                return visit(node.args[-1])
            vals = [visit(arg) for arg in node.args]
            if fname == "abs" and len(vals) == 1:
                return abs(vals[0])
            if fname == "lt" and len(vals) == 2:
                return float(vals[0] < vals[1])
            if fname == "geq" and len(vals) == 2:
                return float(vals[0] >= vals[1])
        raise ValueError(f"Unsupported generated formula AST: {ast.dump(node)}")

    def expression(formula: str) -> float:
        return float(visit(ast.parse(formula, mode="eval").body))

    for name in plan.assignments:
        resolve(name)
    rhs = defaultdict(float)
    for reaction in plan.reactions:
        rate = expression(reaction.formula)
        if not np.isfinite(rate):
            raise ValueError(f"Nonfinite rate for {reaction.id}")
        for name in reaction.reactants:
            rhs[name] -= rate
        for name in reaction.products:
            rhs[name] += rate
    out = np.zeros_like(y)
    for i, gene in enumerate(plan.genes):
        off = int(idx.offset_y[i])
        out[off] = rhs[sid("R", gene)]
        out[off+1] = rhs[sid("P", gene)]
        for j, site in enumerate(plan.sites[gene]):
            out[off+2+j] = rhs[sid("X", gene, site)]
    return out


def verify_rhs(plan: ExportPlan, sys_obj, fitted: dict, *, reference_backend: str = "legacy") -> dict[str, Any]:
    """Compare exported reaction expressions against a PhosKinTime RHS.

    Production uses the JAX/Diffrax fitting RHS from backend.py, not the
    legacy System.rhs path, whose TF-input squashing can differ.
    Synthetic unit tests can explicitly exercise legacy System.rhs.
    """
    if reference_backend not in {"jax", "legacy"}:
        raise ValueError(f"Unsupported reference backend: {reference_backend}")
    if reference_backend == "jax":
        import jax
        import jax.numpy as jnp
        from networkmodel.backend import make_networkmodel_rhs
        jax.config.update("jax_enable_x64", True)
        reference_fun = make_networkmodel_rhs(sys_obj)
        reference_params = {name: jnp.asarray(value, dtype=jnp.float64)
                            for name, value in fitted.items()}
    else:
        reference_fun = None
    grid = np.asarray(sys_obj.kin.grid, dtype=float)
    times = [float(grid[0])]
    if len(grid) > 1:
        times += [float((grid[0]+grid[1])/2), float(grid[1])]
    y0 = np.asarray(sys_obj.y0(), dtype=float)
    # Positive interior test states, not just the potentially sparse initial state.
    y = np.maximum(y0, 0.0) + 0.123
    max_err = 0.0
    for t in times:
        if reference_backend == "jax":
            ref = np.asarray(reference_fun(
                jnp.asarray(t, dtype=jnp.float64),
                jnp.asarray(y, dtype=jnp.float64), reference_params,
            ), dtype=float)
        else:
            ref = np.asarray(sys_obj.rhs(float(t), y), dtype=float)
        alt = evaluate_reaction_plan_rhs(plan, sys_obj, float(t), y)
        err = float(np.max(np.abs(ref-alt)))
        max_err = max(max_err, err)
        if not np.allclose(ref, alt, rtol=1e-8, atol=1e-9):
            bad = np.argsort(np.abs(ref-alt))[-5:].tolist()
            raise RuntimeError(f"Generated SBML reaction/assignment equations differ from PhosKinTime {reference_backend} RHS at t={t}: "
                               f"max_abs_error={err}; state_indices={bad}")
    return {"tested_times": times, "max_abs_rhs_error": max_err,
            "reference_backend": reference_backend}


def _sbml_check(rc, label: str):
    if rc != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(f"libSBML operation failed for {label}, code={rc}")


def write_sbml(plan: ExportPlan, xml_path: Path) -> dict[str, Any]:
    if libsbml is None:
        raise RuntimeError("Install python-libsbml in your PhosKinTime environment: uv pip install python-libsbml")
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    _sbml_check(model.setId("PhosKinTime_fitted_distributive"), "model ID")
    _sbml_check(model.setName("PhosKinTime fitted distributive signalling network"), "model name")
    model.setTimeUnits("dimensionless")
    model.setSubstanceUnits("dimensionless")
    model.setExtentUnits("dimensionless")
    if plan.notes:
        body = '<body xmlns="http://www.w3.org/1999/xhtml">' + "".join(
            f"<p>{xml_escape(text)}</p>" for text in plan.notes
        ) + '</body>'
        _sbml_check(model.setNotes(body), "model notes")
    comp = model.createCompartment()
    comp.setId("cell")
    comp.setConstant(True)
    comp.setSpatialDimensions(0.0)
    comp.setSize(1.0)
    comp.setUnits("dimensionless")

    assigned = set(plan.assignments)
    for name, initial in plan.species.items():
        sp = model.createSpecies()
        sp.setId(name)
        sp.setName(name.replace("_", " "))
        sp.setCompartment("cell")
        sp.setHasOnlySubstanceUnits(True)
        sp.setSubstanceUnits("dimensionless")
        sp.setConstant(False)
        sp.setBoundaryCondition(initial is None)
        if initial is not None:
            sp.setInitialAmount(float(initial))
    for name, value in plan.parameters.items():
        p = model.createParameter()
        p.setId(name)
        p.setName(name)
        p.setConstant(name not in assigned)
        p.setUnits("dimensionless")
        if value is not None:
            p.setValue(float(value))

    for target, formula in plan.assignments.items():
        ast = libsbml.parseL3Formula(formula)
        if ast is None:
            raise ValueError(f"Could not parse assignment for {target}: {formula}")
        rule = model.createAssignmentRule()
        rule.setVariable(target)
        _sbml_check(rule.setMath(ast), f"assignment {target}")

    for r in plan.reactions:
        rx = model.createReaction()
        rx.setId(r.id)
        rx.setName(r.name)
        rx.setReversible(r.reversible)
        for name in r.reactants:
            ref = rx.createReactant()
            ref.setSpecies(name)
            ref.setStoichiometry(1.0)
            ref.setConstant(True)
        for name in r.products:
            ref = rx.createProduct()
            ref.setSpecies(name)
            ref.setStoichiometry(1.0)
            ref.setConstant(True)
        for name in r.modifiers:
            ref = rx.createModifier()
            ref.setSpecies(name)
        ast = libsbml.parseL3Formula(r.formula)
        if ast is None:
            raise ValueError(f"Could not parse kinetic law for {r.id}: {r.formula}")
        kinetic = rx.createKineticLaw()
        _sbml_check(kinetic.setMath(ast), f"kinetic law {r.id}")

    assert model.getNumReactions() == len(plan.reactions)
    assert model.getNumSpecies() == len(plan.species)
    assert model.getNumRules() == len(plan.assignments)
    messages = []
    n = doc.checkConsistency()
    errors = 0
    for i in range(n):
        e = doc.getError(i)
        entry = {"severity": int(e.getSeverity()), "id": int(e.getErrorId()),
                 "message": str(e.getMessage())}
        messages.append(entry)
        if e.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            errors += 1
    if errors:
        raise RuntimeError(f"SBML validation failed ({errors} errors), first: {messages[:5]}")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ok = libsbml.writeSBMLToFile(doc, str(xml_path))
    if not ok:
        raise IOError(f"libSBML did not write {xml_path}")
    # Reopen the on-disk artefact, not just the in-memory document.
    reread = libsbml.readSBMLFromFile(str(xml_path))
    if reread.getModel() is None or reread.getModel().getNumReactions() != len(plan.reactions):
        raise RuntimeError("Roundtrip SBML reaction count mismatch")
    return {"path": str(xml_path), "errors": errors, "warnings_or_info": len(messages),
            "messages": messages}


def metadata(plan: ExportPlan, fitting: dict, result_dir: Path, verification: dict, sbml_validation=None):
    return {"source_directory": str(result_dir), "source_fit": str(result_dir/"fitted_params_picked.json"),
            "model": "networkmodel/distributive", "network": plan.counts,
            "rhs_comparison": verification, "fit_parameter_groups": sorted(fitting),
            "SBML_validation": sbml_validation, "transcription_modifier_edges":
            sum(len(r.modifiers) for r in plan.reactions if r.name.startswith("transcription")),
            "phosphorylation_modifier_edges":
            sum(len(r.modifiers) for r in plan.reactions if r.name.startswith("phosphorylation"))}


def rebuild_system_without_posterior_payload():
    """Rebuild the original topology from the selected PhosKinTime TOML inputs.

    Only topology, kinase inputs and the initial states are reconstructed. Fitted
    physical parameters are loaded separately from fitted_params_picked.json.
    Later strict checks must pass against the fitted result exports.
    """
    from networkmodel import config as cfg
    from networkmodel.io import load_data
    from networkmodel.PosteriorObjective import _prepare_tf_model
    from networkmodel.BuildMatrix import build_W_parallel, build_tf_matrix
    from networkmodel.network import KinaseInput, System
    from networkmodel.utils import normalize_fc_to_t0

    inputs = SimpleNamespace(
        kinase_net=cfg.KINASE_NET_FILE,
        tf_net=cfg.TF_NET_FILE,
        ms=cfg.MS_DATA_FILE,
        rna=cfg.RNA_DATA_FILE,
        phospho=cfg.PHOSPHO_DATA_FILE,
        kinopt=cfg.KINOPT_RESULTS_FILE,
        tfopt=cfg.TFOPT_RESULTS_FILE,
    )
    (df_kin, df_tf, df_prot, df_pho, df_rna,
     kin_beta_map, tf_beta_map) = load_data(inputs)

    if cfg.NORMALIZE_FC_STEADY:
        df_prot = normalize_fc_to_t0(df_prot)
        df_pho = normalize_fc_to_t0(df_pho)
    df_prot_raw = df_prot.copy()

    # Match PosteriorObjective's input normalization and filtering exactly.
    for df in (df_kin, df_pho):
        df["protein"] = df["protein"].astype(str).str.strip()
    df_kin["psite"] = df_kin["psite"].astype(str).str.strip()
    df_pho["psite"] = df_pho["psite"].astype(str).str.strip()
    pairs = set(zip(df_kin["protein"], df_kin["psite"]))
    mask = [(str(p), str(site)) in pairs
            for p, site in zip(df_pho["protein"], df_pho["psite"])]
    df_pho = df_pho.loc[mask].copy()

    df_tf_model, idx = _prepare_tf_model(
        df_tf=df_tf, df_kin=df_kin, df_prot=df_prot, df_rna=df_rna,
        df_pho=df_pho, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map,
    )
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    df_prot_kin = df_prot_raw[df_prot_raw["protein"].isin(idx.kinases)].copy()
    kin_in = KinaseInput(idx.kinases, df_prot_kin)
    W_global = build_W_parallel(df_kin, idx, n_cores=1)
    tf_mat = build_tf_matrix(
        df_tf_model, idx,
        tf_beta_map=tf_beta_map, kin_beta_map=kin_beta_map,
    )
    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(float)
    tf_deg[tf_deg < 1e-12] = 1.0
    defaults = {
        "c_k": np.array([max(0.01, float(kin_beta_map.get(k, 1.0)))
                         for k in idx.kinases]),
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.05),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.1),
        "E_i": np.ones(idx.N),
        "tf_scale": 1.0,
    }
    sys_obj = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)
    if cfg.USE_INITIAL_CONDITION_FROM_DATA:
        sys_obj.attach_initial_condition_data(
            df_prot=df_prot, df_rna=df_rna, df_pho=df_pho,
        )
        sys_obj.set_initial_conditions()
    return sys_obj



class SavedKinaseInput:
    """Time-series boundary input reconstructed from the original matrix export.

    Its stepwise evaluation matches networkmodel.network.KinaseInput.eval, but
    uses the *saved run's* time grid, not whatever config.toml now specifies.
    """

    def __init__(self, grid, matrix):
        self.grid = np.asarray(grid, dtype=float)
        self.Kmat = np.asarray(matrix, dtype=float)
        if self.grid.ndim != 1 or not len(self.grid) or not np.all(np.isfinite(self.grid)):
            raise ValueError("Invalid saved kinase input time grid")
        if np.any(np.diff(self.grid) <= 0):
            raise ValueError("Saved kinase time grid must be strictly increasing")
        if self.Kmat.ndim != 2 or self.Kmat.shape[1] != len(self.grid) or not np.all(np.isfinite(self.Kmat)):
            raise ValueError("Invalid saved kinase input matrix")

    def eval(self, t):
        j = int(np.searchsorted(self.grid, float(t), side="right") - 1)
        return self.Kmat[:, max(0, min(j, len(self.grid)-1))]


def rebuild_system_from_saved_exports(results: Path):
    """Reconstruct the exact *exported* distributive topology without config input paths.

    This uses PhosKinTime's original network matrix exports, protein/site/kinase
    parameter-table ordering, stored initial states and recorded kinase input
    trajectories. It fails on missing/ambiguous data; no zero-valued edges,
    invented sites or default initial values are inserted.

    Subsequent checks compare the output with saved fitted parameters and
    phosphorylation-drive time series. This is independent of the current
    config.toml's data paths, which may have changed after optimization.
    """
    from scipy.sparse import coo_matrix
    from networkmodel.network import System

    genes_df = _read(results / "model_parameters_genes.csv")
    kinases_df = _read(results / "model_parameters_kinases.csv")
    site_df = _read(results / "model_parameters_genes_psites.csv")
    w_df = _read(results / "network_W_global.csv")
    s_df = _read(results / "S_rates_picked.csv")
    tf_df = _read(results / "network_tf_mat.csv")
    ki_df = _read(results / "network_kinase_inputs.csv")
    ic_df = _read(results / "initial_conditions_y0.csv")

    def ordered_names(df, col, label):
        if col not in df.columns:
            raise ValueError(f"{label}: missing column {col!r}")
        vals = [str(v).strip() for v in df[col]]
        if not vals or any(not v or v.lower() == "nan" for v in vals) or len(vals) != len(set(vals)):
            raise ValueError(f"{label}: empty or duplicate identifiers")
        if vals != sorted(vals):
            raise ValueError(f"{label} rows are not in PhosKinTime's sorted Index order")
        return vals

    genes = ordered_names(genes_df, "Protein_Gene", "Saved genes")
    kinases = ordered_names(kinases_df, "Kinase", "Saved kinases")
    p2i = {p: i for i, p in enumerate(genes)}
    k2i = {k: i for i, k in enumerate(kinases)}

    # S_rates_picked.csv is emitted in exact idx.proteins / idx.sites order,
    # and includes zero-drive sites that might not appear in a sparse COO W
    # export. It is the authoritative site-index layout for this fitted run.
    required_s = {"protein", "psite", "time", "S"}
    if not required_s <= set(s_df.columns) or s_df.empty:
        raise ValueError("Saved S_rates_picked.csv is missing complete long-format site drives")
    site_row_keys = list(dict.fromkeys(
        (str(row.protein), str(row.psite)) for row in s_df.itertuples(index=False)
    ))
    sites = {p: [] for p in genes}
    for gene, site in site_row_keys:
        if gene not in sites or not site or site.lower() == "nan":
            raise ValueError(f"Unknown gene or site in saved S rates: {(gene, site)}")
        sites[gene].append(site)
    if site_row_keys != [(g, site) for g in genes for site in sites[g]]:
        raise ValueError("Saved S rates are not ordered by fitted gene blocks")

    required_w = {"Site", "Kinase", "Weight"}
    if not required_w <= set(w_df.columns) or len(w_df) == 0:
        raise ValueError(f"Saved W network lacks columns {required_w} or edges")

    site_file_pairs = set()
    required_ps = {"Protein_Gene", "psite", "Phospho_Degradation_Dp"}
    if not required_ps <= set(site_df.columns):
        raise ValueError(f"Saved phosphosite parameter table missing {required_ps - set(site_df.columns)}")
    for row in site_df.dropna(subset=["psite"]).itertuples(index=False):
        key = (str(row.Protein_Gene), str(row.psite))
        if key in site_file_pairs:
            raise ValueError(f"Duplicate saved site parameter: {key}")
        site_file_pairs.add(key)
    if set(site_row_keys) != site_file_pairs:
        raise ValueError(
            "Saved phosphorylation-rate sites differ from fitted phosphosite table; "
            f"S-only={sorted(set(site_row_keys)-site_file_pairs)[:6]}, "
            f"params-only={sorted(site_file_pairs-set(site_row_keys))[:6]}"
        )

    offset_y = np.zeros(len(genes), dtype=np.int32)
    offset_s = np.zeros(len(genes), dtype=np.int32)
    curr_y = curr_s = 0
    for i, gene in enumerate(genes):
        offset_y[i], offset_s[i] = curr_y, curr_s
        curr_y += 2 + len(sites[gene])
        curr_s += len(sites[gene])
    site2i = {key: i for i, key in enumerate(site_row_keys)}

    w_rows = []
    w_cols = []
    w_values = []
    seen_w = set()
    for row in w_df.itertuples(index=False):
        label = str(row.Site)
        matches = [g for g in genes if label.startswith(g + "_") and (g, label[len(g)+1:]) in site2i]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map saved W Site={label!r} to a fitted phosphosite")
        gene = matches[0]
        site = label[len(gene)+1:]
        kinase = str(row.Kinase)
        key = (gene, site, kinase)
        if kinase not in k2i or key in seen_w:
            raise ValueError(f"Invalid or duplicate kinase-site edge {key}")
        seen_w.add(key)
        val = float(row.Weight)
        if not math.isfinite(val):
            raise ValueError(f"Nonfinite kinase-site weight {key}")
        w_rows.append(site2i[(gene, site)])
        w_cols.append(k2i[kinase])
        w_values.append(val)
    W = coo_matrix((w_values, (w_rows, w_cols)), shape=(curr_s, len(kinases))).tocsr()

    required_tf = {"Target", "Source_TF", "Weight"}
    if not required_tf <= set(tf_df.columns):
        raise ValueError(f"Saved TF matrix missing {required_tf-set(tf_df.columns)}")
    tf_rows = []
    tf_cols = []
    tf_values = []
    seen_tf = set()
    for row in tf_df.itertuples(index=False):
        target, source = str(row.Target), str(row.Source_TF)
        key = (target, source)
        if target not in p2i or source not in p2i or key in seen_tf:
            raise ValueError(f"Saved TF network has unknown/duplicate edge {key}")
        seen_tf.add(key)
        val = float(row.Weight)
        if not math.isfinite(val):
            raise ValueError(f"Nonfinite TF weight {key}")
        tf_rows.append(p2i[target])
        tf_cols.append(p2i[source])
        tf_values.append(val)
    TF = coo_matrix((tf_values, (tf_rows, tf_cols)), shape=(len(genes), len(genes))).tocsr()
    tf_deg = np.asarray(np.abs(TF).sum(axis=1)).ravel().astype(float)
    tf_deg[tf_deg < 1e-12] = 1.0

    # Exact table schema created by runner.py: DataFrame.to_csv(index=True)
    # has its first column named 'Unnamed: 0' on read.
    id_columns = [c for c in ki_df.columns if set(ki_df[c].astype(str)) == set(kinases)]
    if len(id_columns) != 1:
        raise ValueError(f"Cannot uniquely identify kinase names in saved input matrix; candidate columns={id_columns}")
    idcol = id_columns[0]
    if ki_df[idcol].astype(str).duplicated().any():
        raise ValueError("Saved kinase input matrix has duplicate kinase labels")
    tcols = [col for col in ki_df.columns if col != idcol]
    if not tcols or any(not str(c).startswith("t_") for c in tcols):
        raise ValueError(f"Expected t_<time> columns in network_kinase_inputs.csv; got {tcols[:6]}")
    try:
        grid = np.asarray([float(str(c)[2:]) for c in tcols], dtype=float)
    except ValueError as exc:
        raise ValueError("Invalid numeric time columns in saved kinase input matrix") from exc
    kin_rows = ki_df.set_index(idcol).loc[kinases, tcols]
    kmat = kin_rows.to_numpy(dtype=float)
    kin = SavedKinaseInput(grid, kmat)

    required_ic = {"entity", "kind", "substate", "value"}
    if not required_ic <= set(ic_df.columns):
        raise ValueError(f"Saved initial conditions missing {required_ic-set(ic_df.columns)}")
    y0_map = {}
    for row in ic_df.itertuples(index=False):
        key = (str(row.entity), str(row.kind), str(row.substate))
        if key in y0_map:
            raise ValueError(f"Duplicate initial-state entry: {key}")
        y0_map[key] = float(row.value)
    y0 = np.empty(curr_y, dtype=float)
    for i, gene in enumerate(genes):
        off = int(offset_y[i])
        for position, key in [(off, (gene, "mRNA", "R")),
                              (off+1, (gene, "protein_state", "P0"))]:
            if key not in y0_map:
                raise ValueError(f"Missing saved initial value {key}")
            y0[position] = y0_map[key]
        for j, site in enumerate(sites[gene]):
            key = (gene, "phospho_state", site)
            if key not in y0_map:
                raise ValueError(f"Missing saved initial value {key}")
            y0[off+2+j] = y0_map[key]
    if not np.all(np.isfinite(y0)) or np.any(y0 < 0):
        raise ValueError("Initial state contains nonfinite or negative values")

    idx = SimpleNamespace(
        proteins=genes, kinases=kinases, sites=[sites[g] for g in genes],
        p2i=p2i, k2i=k2i, proxy_map={}, N=len(genes),
        n_sites=np.asarray([len(sites[g]) for g in genes], dtype=np.int32),
        offset_y=offset_y, offset_s=offset_s, state_dim=curr_y, total_sites=curr_s,
    )
    defaults = dict(
        c_k=np.ones(len(kinases)), A_i=np.ones(len(genes)),
        B_i=np.ones(len(genes)), C_i=np.ones(len(genes)),
        D_i=np.ones(len(genes)), E_i=np.ones(len(genes)),
        Dp_i=np.ones(curr_s), tf_scale=1.,
    )
    system = System(idx, W, TF, kin, defaults, tf_deg)
    system.custom_y0 = np.asarray(y0, dtype=float)
    return system

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--conf", type=Path, default=Path("config.toml"), help="Original TOML networkmodel configuration")
    ap.add_argument("--run-config", type=Path, default=None, help="Optional explicit posterior_run_config.json, if retained")
    ap.add_argument("--check-only", action="store_true", help="Reconstruct, validate tables and RHS; do not write SBML")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacement of existing output file")
    args = ap.parse_args(argv)
    results = args.results.resolve()
    output_dir = (args.out if args.out else results/"sbml_reactions").resolve()
    xml_path = output_dir/"networkmodel_reactions.xml"
    if not results.is_dir():
        ap.error(f"Results directory does not exist: {results}")
    if not args.check_only and xml_path.exists() and not args.overwrite:
        ap.error(f"Refusing to overwrite {xml_path}; use --overwrite explicitly")
    if not args.conf.exists():
        ap.error(f"Missing config TOML: {args.conf}")
    os.environ["PHOSKINTIME_NETWORKMODEL_CONFIG"] = str(args.conf.resolve())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    # Only import after setting the config variable; networkmodel.config is import-time configured.
    from networkmodel.config import MODEL
    if MODEL != 0:
        raise RuntimeError(f"Only distributive MODEL=0 is implemented; loaded TOML selected MODEL={MODEL}. "
                           "Use the original distributive configuration; never silently export a different topology.")
    cfg_path = (args.run_config.resolve() if args.run_config is not None
                else results / "posterior_payload" / "posterior_run_config.json")
    if args.run_config is not None and not cfg_path.is_file():
        raise FileNotFoundError(f"Explicit --run-config file does not exist: {cfg_path}")
    # Saved run exports are authoritative for the fitted network; the current
    # config.toml may point to a different (e.g. two-gene) example network.
    # Prefer reconstruction from *saved* complete model matrices, with strict
    # param/trajectory/ODE verification below. No rerun of optimization.
    print("Rebuilding fitted topology from saved network matrices, inputs, and y0.")
    if cfg_path.is_file():
        print(f"Posterior manifest found ({cfg_path}); not required for SBML export.")
    sys_obj = rebuild_system_from_saved_exports(results)
    idx = sys_obj.idx
    fitted = read_fitted_params(results, idx)
    sys_obj.update(**fitted)
    count_edges = check_rebuilt_network(results, sys_obj)
    check_fitted_parameter_tables(results, idx, fitted)
    n_verified_rates = check_phospho_drive_samples(results, sys_obj, fitted)
    n_verified_ic = check_initial_conditions_table(results, sys_obj)
    plan = make_plan(sys_obj, fitted)
    verified = verify_rhs(plan, sys_obj, fitted, reference_backend="jax")
    verified.update(count_edges)
    verified["phosphorylation_drive_rows_checked"] = n_verified_rates
    verified["initial_state_rows_checked"] = n_verified_ic
    print(json.dumps(metadata(plan, fitted, results, verified), indent=2))
    if args.check_only:
        print("Preflight PASS. No SBML written.")
        return 0
    validation = write_sbml(plan, xml_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"networkmodel_sbml_manifest.json").write_text(
        json.dumps(metadata(plan, fitted, results, verified, validation), indent=2))
    # Separate graphical edge manifest for Cytoscape cross-check.
    edges=[]
    for r in plan.reactions:
        for role, names in (("reactant", r.reactants), ("product", r.products),
                            ("modifier", r.modifiers)):
            for name in names:
                edges.append({"reaction": r.id, "species": name, "role": role,
                              "kinetic_law": r.formula})
    pd.DataFrame(edges, columns=["reaction", "species", "role", "kinetic_law"]).to_csv(
        output_dir/"networkmodel_reaction_graph_edges.csv", index=False)
    print(f"SBML: {xml_path}\nReactions: {len(plan.reactions)} | Species: {len(plan.species)}")
    print(f"Manifest: {output_dir/'networkmodel_sbml_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())