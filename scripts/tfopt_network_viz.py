#!/usr/bin/env python3
"""
tfopt_network_viz.py

Publication-style, minimal visualization for TFopt network readouts produced by tfopt_network_readout.py.

Inputs (CSV, from tfopt_network_readout.py):
  - results_scripts/figures_tfopt/tfopt_tf_load.csv
  - results_scripts/figures_tfopt/tfopt_target_dominant_tfs.csv
  - results_scripts/figures_tfopt/tfopt_knockout_effects.csv
  - results_scripts/figures_tfopt/tfopt_tf_psite_stats.csv

Optional:
  - data/input4.csv (TF->mRNA edges) for a network plot

Outputs (PNG + PDF):
  - figures_tfopt_viz/*.png
  - figures_tfopt_viz/*.pdf

Notes:
  - Minimal titles, publication defaults.
  - No seaborn. Matplotlib only.
  - If you want exact journal styling (fonts, tick sizes), edit STYLE section.

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.integrate import trapezoid

# -----------------------
# CONFIG
# -----------------------
IN_DIR = Path("./results_scripts/figures_tfopt")  # where readout CSVs are written
OUT_DIR = Path("./results_scripts/figures_tfopt")  # where figures are written
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_CSV = Path("data/input4.csv")  # optional (Source,Target)

# tfopt mRNA time grid
MRNA_TIME_POINTS = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960], dtype=float)

# labeling controls
MAX_POINT_LABELS = 5  # for scatter outliers
TOP_TF_BAR = 25  # top TFs in load bar
TOP_TARGETS = 25  # top targets in dominance display
TOP_KO_PER_TARGET = 8  # TFs shown per target in KO heatmap

# scatter outlier rule (95% band): label top-N by perpendicular distance to identity line
LABEL_BY = "perp_dist"  # abs(pred-obs)

# file formats
SAVE_PNG_DPI = 300
SAVE_PDF = False


# -----------------------
# STYLE
# -----------------------
def style_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": SAVE_PNG_DPI,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(stem: Path):
    # Always save PNG; optionally also PDF
    plt.tight_layout()
    plt.savefig(str(stem) + ".png", bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close()


def light_grid(ax=None):
    ax = ax or plt.gca()
    ax.grid(alpha=0.25, linewidth=0.6)


# -----------------------
# LOAD
# -----------------------
def load_inputs(in_dir: Path):
    tf_load = pd.read_csv(in_dir / "tfopt_tf_load.csv")
    dom = pd.read_csv(in_dir / "tfopt_target_dominant_tfs.csv")
    ko = pd.read_csv(in_dir / "tfopt_knockout_effects.csv")
    pstat = pd.read_csv(in_dir / "tfopt_tf_psite_stats.csv")
    return tf_load, dom, ko, pstat


# -----------------------
# PLOTS
# -----------------------
def plot_tf_total_load(tf_load: pd.DataFrame):
    d = tf_load.copy()
    d = d.sort_values("total_load_auc_abs", ascending=False).head(TOP_TF_BAR)
    d = d.iloc[::-1]  # for barh bottom-up

    plt.figure(figsize=(6.0, max(3.4, 0.22 * len(d) + 1.2)))
    y = np.arange(len(d))
    plt.barh(y, d["total_load_auc_abs"].to_numpy(dtype=float))
    plt.yticks(y, d["TF"].astype(str).to_list())
    plt.xlabel("Total control load")
    light_grid()
    savefig(OUT_DIR / "tf_total_load")


def plot_tf_beta_bound_vs_load(tf_load: pd.DataFrame):
    d = tf_load.dropna(subset=["total_load_auc_abs", "frac_beta_at_bound"]).copy()
    x = d["total_load_auc_abs"].astype(float).to_numpy()
    y = d["frac_beta_at_bound"].astype(float).to_numpy()

    # label a few extremes: top load and high bound pressure
    d["score_label"] = (d["total_load_auc_abs"].rank(pct=True) + d["frac_beta_at_bound"].rank(pct=True))
    lab = d.sort_values("score_label", ascending=False).head(12)

    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(x, y, s=26, alpha=0.85)
    for _, r in lab.iterrows():
        plt.annotate(r["TF"], (r["total_load_auc_abs"], r["frac_beta_at_bound"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points", alpha=0.9)
    plt.xlabel("Total control load")
    plt.ylabel("Fraction β at bound")
    light_grid()
    savefig(OUT_DIR / "tf_bound_pressure_vs_load")


def plot_dominant_tf_share(dom: pd.DataFrame):
    d = dom.copy()
    d = d.sort_values("dominant_overall_share", ascending=False).head(TOP_TARGETS)
    d = d.iloc[::-1]

    plt.figure(figsize=(6.4, max(3.6, 0.22 * len(d) + 1.2)))
    y = np.arange(len(d))
    plt.barh(y, d["dominant_overall_share"].astype(float).to_numpy())
    plt.yticks(y, (d["mRNA"].astype(str) + "  (" + d["dominant_overall"].astype(str) + ")").to_list())
    plt.xlabel("Dominant TF share")
    plt.xlim(0, 1.0)
    light_grid()
    savefig(OUT_DIR / "target_dominant_tf_share")


def plot_obs_vs_recon(dom: pd.DataFrame):
    """
    Scatter: observed vs reconstructed magnitude.
    Uses dom columns: obs_auc_abs, recon_auc_abs (as written by readout script).
    Labels top deviations from identity among points with non-zero observed.
    """
    d = dom.dropna(subset=["obs_auc_abs", "recon_auc_abs"]).copy()
    d["obs"] = d["obs_auc_abs"].astype(float)
    d["recon"] = d["recon_auc_abs"].astype(float)
    d = d[(d["obs"].abs() > 1e-12) | (d["recon"].abs() > 1e-12)].copy()

    # non-zero observed only (as before)
    d_nz = d[d["obs"].abs() > 1e-12].copy()

    # distance to identity line (perpendicular; identity is y=x)
    d_nz["perp_dist"] = np.abs(d_nz["recon"] - d_nz["obs"]) / np.sqrt(2)

    # 95% band threshold (around the identity line, not a fit to the data)
    thr = np.quantile(d_nz["perp_dist"].to_numpy(), 0.95)

    # label only points that fall within the 95% parallel band to identity
    in_band = d_nz[d_nz["perp_dist"] <= thr].copy()

    # optionally cap how many labels (pick closest to identity so labels are "most within" band)
    lab = in_band.sort_values("perp_dist", ascending=True).head(MAX_POINT_LABELS)

    # --- identity-centered scatter (no negative axis shown) ---
    x = d["obs"].to_numpy()
    y = d["recon"].to_numpy()

    mx = float(np.nanmax(np.r_[x, y, 1e-9]))

    plt.figure(figsize=(5.2, 4.1))
    plt.scatter(x, y, s=22, alpha=0.85)

    # identity line through origin
    xline = np.array([0.0, mx])
    plt.plot(xline, xline, linewidth=1.0, color="black")

    # optional: 95% band around identity (computed from data distances)
    d_nz = d[d["obs"].abs() > 1e-12].copy()
    perp = np.abs(d_nz["recon"] - d_nz["obs"]) / np.sqrt(2)
    thr = np.quantile(perp.to_numpy(), 0.95)
    offset = thr * np.sqrt(2)

    plt.plot(xline, xline + offset, linewidth=0.8, alpha=0.5, color="black")
    plt.plot(xline, xline - offset, linewidth=0.8, alpha=0.5, color="black")

    # force axes to show only non-negative region and include origin
    plt.xlim(0.0, mx)
    plt.ylim(0.0, mx)

    # keep aspect so identity is 45 degrees
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()

    for _, r in lab.iterrows():
        plt.annotate(
            str(r["mRNA"]),
            (r["obs"], r["recon"]),
            fontsize=5,
            xytext=(3, 3),
            textcoords="offset fontsize",
            alpha=0.9,
        )

    plt.xlabel("Observed magnitude")
    plt.ylabel("Reconstructed magnitude")
    light_grid()
    savefig(OUT_DIR / "obs_vs_recon_magnitude")


def plot_ko_heatmap_top_targets(ko: pd.DataFrame):
    """
    Heatmap: for top targets (by baseline magnitude), show top KO effects (delta_auc_abs) per target.
    This is interpretable as "which TF removal changes the reconstructed target most".
    """
    d = ko.dropna(subset=["mRNA", "KnockedTF", "delta_auc_abs"]).copy()
    d["delta"] = d["delta_auc_abs"].astype(float)

    # pick targets with largest baseline signal
    base = d.groupby("mRNA", as_index=False)["baseline_auc_abs"].max()
    base = base.sort_values("baseline_auc_abs", ascending=False).head(min(18, base.shape[0]))
    targets = base["mRNA"].astype(str).to_list()

    # for each target, select top K TFs by |delta|
    rows = []
    all_tfs = set()
    for t in targets:
        dt = d[d["mRNA"].astype(str) == t].copy()
        dt["absd"] = dt["delta"].abs()
        dt = dt.sort_values("absd", ascending=False).head(TOP_KO_PER_TARGET)
        rows.append(dt)
        all_tfs.update(dt["KnockedTF"].astype(str).to_list())

    if not rows:
        return

    sub = pd.concat(rows, ignore_index=True)
    tfs = sorted(all_tfs)

    # matrix: targets x TFs
    M = np.zeros((len(targets), len(tfs)), dtype=float)
    for i, t in enumerate(targets):
        dt = sub[sub["mRNA"].astype(str) == t]
        for _, r in dt.iterrows():
            j = tfs.index(str(r["KnockedTF"]))
            M[i, j] = float(r["delta"])

    plt.figure(figsize=(min(10.5, 0.42 * len(tfs) + 3.2), min(8.0, 0.36 * len(targets) + 2.4)))
    im = plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.035, pad=0.02, label="KO effect (Δ magnitude)")

    plt.yticks(np.arange(len(targets)), targets)
    plt.xticks(np.arange(len(tfs)), tfs, rotation=90)

    plt.xlabel("TF")
    plt.ylabel("mRNA")
    savefig(OUT_DIR / "ko_effects_heatmap")


def plot_network_overview(edges_csv: Path, tf_load: pd.DataFrame, dom: pd.DataFrame):
    """
    Simple bipartite network plot:
      TF -> mRNA edges.
    Node size encodes TF load for TFs and observed magnitude for targets (if available).
    """
    if not edges_csv.exists():
        return

    edges = pd.read_csv(edges_csv)
    if not {"Source", "Target"}.issubset(edges.columns):
        return

    edges["Source"] = edges["Source"].astype(str)
    edges["Target"] = edges["Target"].astype(str)

    # focus on a tractable subgraph: top TFs by load + top targets by observed magnitude
    top_tfs = set(tf_load.sort_values("total_load_auc_abs", ascending=False).head(20)["TF"].astype(str).to_list())
    top_targets = set(dom.sort_values("obs_auc_abs", ascending=False).head(25)["mRNA"].astype(str).to_list())

    e = edges[(edges["Source"].isin(top_tfs)) & (edges["Target"].isin(top_targets))].copy()
    if e.empty:
        # fall back to top TFs only
        e = edges[edges["Source"].isin(top_tfs)].copy()
    if e.empty:
        return

    G = nx.DiGraph()
    for _, r in e.iterrows():
        G.add_edge(r["Source"], r["Target"])

    # sizes
    tf_size = {r["TF"]: float(r.get("total_load_auc_abs", 0.0)) for _, r in tf_load.iterrows()}
    tgt_size = {r["mRNA"]: float(r.get("obs_auc_abs", 0.0)) for _, r in dom.iterrows()}

    tf_nodes = sorted(set(e["Source"]))
    tgt_nodes = sorted(set(e["Target"]))

    # bipartite layout
    pos = {}
    # TFs on left, targets on right
    for i, n in enumerate(tf_nodes):
        pos[n] = (0.0, float(i))
    for i, n in enumerate(tgt_nodes):
        pos[n] = (1.0, float(i))

    # scale sizes
    def _scale(vals, lo=60.0, hi=420.0):
        v = np.array(list(vals), dtype=float)
        if len(v) == 0:
            return []
        vmin, vmax = float(v.min()), float(v.max())
        if vmax - vmin < 1e-12:
            return [0.5 * (lo + hi)] * len(v)
        return list(lo + (hi - lo) * (v - vmin) / (vmax - vmin))

    tf_sizes = _scale([tf_size.get(n, 0.0) for n in tf_nodes], lo=80, hi=520)
    tgt_sizes = _scale([tgt_size.get(n, 0.0) for n in tgt_nodes], lo=80, hi=520)

    plt.figure(figsize=(9.8, 0.24 * max(len(tf_nodes), len(tgt_nodes)) + 3.0))
    nx.draw_networkx_edges(G, pos, arrows=True, width=0.8, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_size=tf_sizes, node_shape="s", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=tgt_nodes, node_size=tgt_sizes, node_shape="o", alpha=0.9)

    # labels: only for readability (limit)
    for n in tf_nodes:
        x, y = pos[n]
        plt.text(x - 0.02, y, n, ha="right", va="center", fontsize=7)
    for n in tgt_nodes:
        x, y = pos[n]
        plt.text(x + 0.02, y, n, ha="left", va="center", fontsize=7)

    plt.axis("off")
    savefig(OUT_DIR / "network_overview_top")


# -----------------------------
# EGFR-centric control logic (TFopt)
# -----------------------------
from networkx.drawing.nx_pydot import to_pydot, graphviz_layout


def graphviz_lr_layout(G: nx.DiGraph) -> dict:
    """
    Guaranteed left-to-right DOT layout.
    Raises if Graphviz fails.
    """
    P = to_pydot(G)

    # Force left-to-right layout
    P.set_rankdir("LR")

    # Optional but helps aesthetics
    P.set_nodesep("0.35")
    P.set_ranksep("0.6")

    # Validate nodes exist
    if not P.get_nodes():
        raise RuntimeError("DOT graph has no nodes — check node IDs.")

    pos = graphviz_layout(G, prog="dot")

    if len(pos) != G.number_of_nodes():
        raise RuntimeError("Graphviz layout incomplete — DOT parsing failed.")

    return pos


def _dot_layout(G: nx.DiGraph) -> dict:
    """
    Try Graphviz DOT layout (via pydot). Fall back if unavailable.
    """
    try:
        from networkx.drawing.nx_pydot import graphviz_layout  # type: ignore
        pos = graphviz_layout(G, prog="dot")
        if isinstance(pos, dict) and len(pos) == G.number_of_nodes():
            return pos
    except Exception:
        pass

    # Fallback: TFs on left, target on right
    tfs = [n for n, d in G.nodes(data=True) if d.get("kind") == "tf"]
    tgt = [n for n, d in G.nodes(data=True) if d.get("kind") == "target"]
    pos = {}
    for i, n in enumerate(sorted(tfs)):
        pos[n] = (0.0, float(i))
    for i, n in enumerate(sorted(tgt)):
        pos[n] = (1.0, float(i))
    return pos


def plot_egfr_control_logic_tfopt(
        ko: pd.DataFrame,
        target: str = "EGFR",
        topk: int = 70,
):
    """
    Build EGFR control logic from KO effects:
      - Nodes: TFs + target node
      - Edges: TF -> target
      - Edge width: |delta_auc_abs|
      - Edge linestyle: solid if delta_auc_abs>0 (supports signal), dashed if <0 (suppresses)
    """
    d = ko.copy()
    _require = {"mRNA", "KnockedTF", "delta_auc_abs", "alpha"}
    missing = [c for c in _require if c not in d.columns]
    if missing:
        raise ValueError(f"KO CSV missing {missing}. Have: {list(d.columns)}")

    eg = d[d["mRNA"].astype(str) == target].copy()
    if eg.empty:
        raise ValueError(f"No KO rows found for target={target}.")

    eg["delta"] = eg["delta_auc_abs"].astype(float)
    eg["abs_delta"] = eg["delta"].abs()

    # Keep top-k by |effect|
    eg = eg.sort_values("abs_delta", ascending=False).head(topk).copy()

    # Build graph
    G = nx.DiGraph()
    tgt_id = f"tgt__{target}"
    G.add_node(tgt_id, kind="target", label=target)

    for _, r in eg.iterrows():
        tf = str(r["KnockedTF"])
        tf_id = f"tf__{tf}"
        if tf_id not in G:
            G.add_node(tf_id, kind="tf", label=tf)

        w = float(r["abs_delta"])
        sign = 1 if float(r["delta"]) > 0 else -1
        alpha = float(r.get("alpha", np.nan))
        G.add_edge(tf_id, tgt_id, weight=w, sign=sign, alpha=alpha)

    # pos = _dot_layout(G)

    pos = graphviz_lr_layout(G)

    # Normalize edge widths
    weights = np.array([G.edges[e]["weight"] for e in G.edges], dtype=float)
    wmin, wmax = float(weights.min()), float(weights.max())

    def wscale(w):
        if wmax - wmin < 1e-12:
            return 2.2
        return 0.7 + 3.3 * (w - wmin) / (wmax - wmin)

    # Draw
    plt.figure(figsize=(7.2, max(3.6, 0.25 * (len(G.nodes) + 2))))
    ax = plt.gca()

    tf_nodes = [n for n, dd in G.nodes(data=True) if dd.get("kind") == "tf"]
    tgt_nodes = [n for n, dd in G.nodes(data=True) if dd.get("kind") == "target"]

    # Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_size=260, node_shape="s", alpha=0.95, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=tgt_nodes, node_size=520, node_shape="o", alpha=0.95, ax=ax)

    # Labels
    for n, dd in G.nodes(data=True):
        x, y = pos[n]
        lab = dd.get("label", n)
        if dd.get("kind") == "target":
            plt.text(x, y, lab, ha="center", va="center", fontsize=9)
        else:
            plt.text(x - 0.02, y, lab, ha="right", va="center", fontsize=8)

    # Edges
    for u, v, ed in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ls = "-" if ed["sign"] > 0 else "--"
        lw = wscale(float(ed["weight"]))
        ax.plot([x1, x2], [y1, y2], linewidth=lw, linestyle=ls, alpha=0.85)

    ax.axis("off")
    plt.text(
        0.01, 0.01,
        "Width ∝ |KO effect|  ·  Solid: supports EGFR signal  ·  Dashed: suppresses EGFR signal",
        transform=ax.transAxes, fontsize=7, ha="left", va="bottom"
    )
    savefig(OUT_DIR / f"{target}_control_logic")


def plot_egfr_ko_bar_tfopt(
        ko: pd.DataFrame,
        target: str = "EGFR",
        topk: int = 70,
):
    """
    Horizontal barplot of KO effects (delta_auc_abs) for EGFR.
    Right: supporting TFs; Left: suppressing TFs.
    """
    eg = ko[ko["mRNA"].astype(str) == target].copy()
    if eg.empty:
        raise ValueError(f"No KO rows found for target={target}.")

    eg["delta"] = eg["delta_auc_abs"].astype(float)
    eg["abs_delta"] = eg["delta"].abs()
    eg = eg.sort_values("abs_delta", ascending=False).head(topk).copy()

    eg = eg.iloc[::-1]
    y = np.arange(len(eg))

    plt.figure(figsize=(6.2, max(3.2, 0.22 * len(eg) + 1.1)))
    plt.barh(y, eg["delta"].to_numpy(dtype=float))
    plt.yticks(y, eg["KnockedTF"].astype(str).to_list())
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("KO effect on reconstructed EGFR (Δ AUC|·|)")
    light_grid()
    savefig(OUT_DIR / f"{target}_ko_effects_bar")


def add_egfr_panels(ko: pd.DataFrame):
    plot_egfr_control_logic_tfopt(ko, target="EGFR", topk=18)
    plot_egfr_ko_bar_tfopt(ko, target="EGFR", topk=20)


# -----------------------
# MAIN
# -----------------------
def main():
    style_matplotlib()
    tf_load, dom, ko, pstat = load_inputs(IN_DIR)

    # sanity: accept either old or new column names
    # dominant CSV must contain obs_auc_abs and recon_auc_abs; if you used pred/recon naming, map it.
    dom = dom.copy()
    if "recon_auc_abs" not in dom.columns and "pred_auc_abs" in dom.columns:
        dom = dom.rename(columns={"pred_auc_abs": "recon_auc_abs"})
    if "recon_peak_abs" not in dom.columns and "pred_peak_abs" in dom.columns:
        dom = dom.rename(columns={"pred_peak_abs": "recon_peak_abs"})

    plot_tf_total_load(tf_load)
    plot_tf_beta_bound_vs_load(tf_load)
    plot_dominant_tf_share(dom)
    plot_obs_vs_recon(dom)
    plot_ko_heatmap_top_targets(ko)
    plot_network_overview(EDGES_CSV, tf_load, dom)
    add_egfr_panels(ko)
    print("Wrote figures to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
