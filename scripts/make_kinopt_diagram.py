#!/usr/bin/env python3
"""
Replicate the kinase-optimization schematic (P–S–K–KS) using NetworkX + Matplotlib.

Output: kinopt_diagram.png
"""

from __future__ import annotations
import pydot
import os
import pygraphviz as pgv

def make_kinopt_diagram_dot(
    outfile: str = "kinopt_diagram.png",
    *,
    fmt: str | None = None,
    engine: str = "dot",
    dpi: int = 300,
    rankdir: str = "TB",
    kin_psites: int = 1,              # PSites per kinase: β_{1,k}..β_{kin_psites,k}
):
    """
    Kinopt DOT schematic (consistent with the integrated global diagram):

      P -> S via α
      S -> K via β_{k,s}
      Each kinase K_k aggregates its own:
        - protein component K_k(t) via β_{0,k}
        - PSite components PSite_{p,k}(t) via β_{p,k}, p=1..kin_psites
    """
    import os

    if fmt is None:
        _, ext = os.path.splitext(outfile)
        fmt = ext.lstrip(".").lower() or "png"

    # Palette
    col_P = "#ff1a1a"
    col_S = "#1f77ff"
    col_K = "#7cb518"
    col_inp = "#00a6ff"
    col_alpha = "#7cb518"
    col_beta = "#1f77ff"

    # Config
    S_nodes = ["S1", "S2"]
    K_nodes = ["K1", "K2", "K3", "K4"]
    kin_alpha_edges = [("P", "S1"), ("P", "S2")]
    kin_beta_edges = [("S1", "K1"), ("S1", "K2"), ("S2", "K3"), ("S2", "K4")]

    dot = []
    dot.append("digraph KINOPT {")
    dot.append(
        f'  graph [rankdir={rankdir}, dpi={dpi}, splines=true, bgcolor="white", nodesep=0.55, ranksep=0.85];'
    )
    dot.append('  node  [shape=circle, style=filled, fontname="Helvetica", fontsize=16, fontcolor="black", penwidth=2];')
    dot.append('  edge  [fontname="Helvetica", fontsize=14, fontcolor="black", penwidth=2, arrowsize=0.9];')

    dot.append(f'  P [label=<P<sub>i</sub>>, fillcolor="{col_P}", width=1.1, fixedsize=true];')

    dot.append('  subgraph cluster_kinopt {')
    dot.append('    label=""; color="white"; penwidth=0;')

    for s_idx, s in enumerate(S_nodes, start=1):
        dot.append(f'    {s} [label=<S<sub>{s_idx}</sub>>, fillcolor="{col_S}", width=1.1, fixedsize=true];')

    for k_idx, k in enumerate(K_nodes, start=1):
        dot.append(f'    {k} [label=<K<sub>{k_idx}</sub>>, fillcolor="{col_K}", width=1.1, fixedsize=true];')

    # α: P -> S
    for (_, s) in kin_alpha_edges:
        s_idx = int(s.replace("S", ""))
        dot.append(f'    P -> {s} [color="{col_alpha}", label=<&#945;<sub>{s_idx}</sub>>];')

    # β: S -> K (β_{k,s})
    for (s, k) in kin_beta_edges:
        s_idx = int(s.replace("S", ""))
        k_idx = int(k.replace("K", ""))
        dot.append(f'    {s} -> {k} [color="{col_beta}", label=<&#946;<sub>{k_idx},{s_idx}</sub>>];')

    # Per-kinase β0 + PSites
    for k_idx, k in enumerate(K_nodes, start=1):
        k0 = f"{k}_0"
        dot.append(f'    {k0} [label=<K<sub>{k_idx}</sub>(t)>, fillcolor="{col_inp}", width=1.35, fixedsize=true];')
        dot.append(f'    {k0} -> {k} [color="{col_beta}", label=<&#946;<sub>0,{k_idx}</sub>>];')

        for p in range(1, kin_psites + 1):
            kp = f"{k}_p{p}"
            dot.append(
                f'    {kp} [label=<PSite<sub>{p},{k_idx}</sub>(t)>, fillcolor="{col_inp}", width=1.55, fixedsize=true];'
            )
            dot.append(f'    {kp} -> {k} [color="{col_beta}", label=<&#946;<sub>{p},{k_idx}</sub>>];')

    dot.append('  }')
    dot.append("}")
    dot_str = "\n".join(dot)

    # Render
    try:
        A = pgv.AGraph(string=dot_str)
        A.layout(prog=engine)
        A.draw(outfile, format=fmt)
    except Exception:
        try:
            graphs = pydot.graph_from_dot_data(dot_str)
            if not graphs:
                raise RuntimeError("pydot failed to parse DOT.")
            g = graphs[0]
            write_fn = getattr(g, f"write_{fmt}", None)
            if write_fn is None:
                g.write_raw(outfile)
            else:
                write_fn(outfile, prog=engine)
        except Exception as e:
            raise RuntimeError(
                "Could not render via pygraphviz or pydot. "
                "You likely need Graphviz + one of: pygraphviz, pydot.\n"
                "Install examples:\n"
                "  sudo apt-get install graphviz graphviz-dev\n"
                "  pip install pygraphviz\n"
                "or:\n"
                "  pip install pydot\n"
                f"Original error: {e}"
            )

    print(f"Saved: {outfile}")


def make_tfopt_diagram_dot(
    outfile: str = "tfopt_diagram.png",
    *,
    fmt: str | None = None,
    engine: str = "dot",
    dpi: int = 300,
    rankdir: str = "TB",
    tf_psites: int = 1,               # PSites per TF: β_{1,j}..β_{tf_psites,j}
):
    """
    TFopt DOT schematic:

      TF_j aggregates its own:
        - protein component TF_j(t) via β_{0,j}
        - PSite components PSite_{p,j}(t) via β_{p,j}, p=1..tf_psites
      TF_j -> P via α_{i,j}

    P is the output hub.
    """
    import os

    if fmt is None:
        _, ext = os.path.splitext(outfile)
        fmt = ext.lstrip(".").lower() or "png"

    # Palette
    col_P = "#ff1a1a"
    col_TF = "#1f77ff"
    col_inp = "#00a6ff"
    col_alpha = "#7cb518"
    col_beta = "#1f77ff"

    TFs = [f"TF{j}" for j in range(1, 5)]

    dot = []
    dot.append("digraph TFOPT {")
    dot.append(
        f'  graph [rankdir={rankdir}, dpi={dpi}, splines=true, bgcolor="white", nodesep=0.55, ranksep=0.85];'
    )
    dot.append('  node  [shape=circle, style=filled, fontname="Helvetica", fontsize=16, fontcolor="black", penwidth=2];')
    dot.append('  edge  [fontname="Helvetica", fontsize=14, fontcolor="black", penwidth=2, arrowsize=0.9];')

    dot.append(f'  P [label=<P<sub>i</sub>>, fillcolor="{col_P}", width=1.1, fixedsize=true];')

    dot.append('  subgraph cluster_tfopt {')
    dot.append('    label=""; color="white"; penwidth=0;')

    for j, tf in enumerate(TFs, start=1):
        dot.append(f'    {tf} [label=<TF<sub>{j}</sub>>, fillcolor="{col_TF}", width=1.1, fixedsize=true];')

        tf0 = f"{tf}_0"
        dot.append(f'    {tf0} [label=<TF<sub>{j}</sub>(t)>, fillcolor="{col_inp}", width=1.35, fixedsize=true];')
        dot.append(f'    {tf0} -> {tf} [color="{col_beta}", label=<&#946;<sub>0,{j}</sub>>];')

        for p in range(1, tf_psites + 1):
            ps = f"{tf}_p{p}"
            dot.append(
                f'    {ps} [label=<PSite<sub>{p},{j}</sub>(t)>, fillcolor="{col_inp}", width=1.55, fixedsize=true];'
            )
            dot.append(f'    {ps} -> {tf} [color="{col_beta}", label=<&#946;<sub>{p},{j}</sub>>];')

        dot.append(f'    {tf} -> P [color="{col_alpha}", label=<&#945;<sub>i,{j}</sub>>];')

    dot.append('  }')
    dot.append("}")
    dot_str = "\n".join(dot)

    # Render
    try:
        A = pgv.AGraph(string=dot_str)
        A.layout(prog=engine)
        A.draw(outfile, format=fmt)
    except Exception:
        try:
            graphs = pydot.graph_from_dot_data(dot_str)
            if not graphs:
                raise RuntimeError("pydot failed to parse DOT.")
            g = graphs[0]
            write_fn = getattr(g, f"write_{fmt}", None)
            if write_fn is None:
                g.write_raw(outfile)
            else:
                write_fn(outfile, prog=engine)
        except Exception as e:
            raise RuntimeError(
                "Could not render via pygraphviz or pydot. "
                "You likely need Graphviz + one of: pygraphviz, pydot.\n"
                "Install examples:\n"
                "  sudo apt-get install graphviz graphviz-dev\n"
                "  pip install pygraphviz\n"
                "or:\n"
                "  pip install pydot\n"
                f"Original error: {e}"
            )

    print(f"Saved: {outfile}")

def make_global_diagram_dot(
    outfile: str = "kin_tf_opt_diagram.png",
    *,
    fmt: str | None = None,          # inferred from outfile if None
    engine: str = "dot",             # "dot" (schematic), "neato" (manual-ish), "fdp", "sfdp"
    dpi: int = 300,
    rankdir: str = "TB",             # "TB" (top-bottom) or "LR" (left-right)
    kin_psites: int = 1,             # number of PSites per kinase (β2,k etc.)
    tf_psites: int = 1,              # number of PSites per TF
):
    """
    Single DOT/Graphviz-rendered diagram integrating kinopt + tfopt, with consistent labeling.

    Kinopt (corrected):
      P -> S via α
      S -> K via β_{k,s}
      Each kinase K_k aggregates its own:
        - protein component K_k(t) via β_{0,k}
        - PSite components PSite_{p,k}(t) via β_{p,k}, p=1..kin_psites

    TFopt:
      TF_j aggregates its own:
        - protein component TF_j(t) via β_{0,j}
        - PSite components PSite_{p,j}(t) via β_{p,j}, p=1..tf_psites
      TF_j -> P via α_{i,j}

    Notes:
      - P is a shared node (as you requested).
      - Rendering uses pygraphviz if available, else pydot. Graphviz must be installed.
    """
    if fmt is None:
        _, ext = os.path.splitext(outfile)
        fmt = ext.lstrip(".").lower() or "png"

    # Palette consistent with your previous figures
    col_P = "#ff1a1a"       # red
    col_S = "#1f77ff"       # blue (S nodes / TF nodes)
    col_K = "#7cb518"       # green (K nodes)
    col_inp = "#00a6ff"     # cyan (inputs/PSites)
    col_alpha = "#7cb518"   # green α edges
    col_beta = "#1f77ff"    # blue β edges

    # ---- Kinopt config ----
    S_nodes = ["S1", "S2"]
    K_nodes = ["K1", "K2", "K3", "K4"]

    # P -> S edges (alphas)
    kin_alpha_edges = [("P", "S1"), ("P", "S2")]

    # S -> K edges (betas)
    kin_beta_edges = [("S1", "K1"), ("S1", "K2"), ("S2", "K3"), ("S2", "K4")]

    # ---- TFopt config ----
    TFs = [f"TF{j}" for j in range(1, 3)]

    # ---- Build DOT ----
    dot = []
    dot.append("digraph GLOBAL {")
    dot.append(
        f'  graph [rankdir={rankdir}, dpi={dpi}, splines=true, bgcolor="white", '
        f'nodesep=0.55, ranksep=0.85];'
    )
    dot.append('  node  [shape=circle, style=filled, fontname="Helvetica", fontsize=16, fontcolor="black", penwidth=2];')
    dot.append('  edge  [fontname="Helvetica", fontsize=14, fontcolor="black", penwidth=2, arrowsize=0.9];')

    # Shared hub node P
    dot.append(f'  P [label=<P<sub>i</sub>>, fillcolor="{col_P}", width=1.1, fixedsize=true];')

    # -------------------------
    # Kinopt module
    # -------------------------
    dot.append('  subgraph cluster_kinopt {')
    dot.append('    label=""; color="white"; penwidth=0;')

    # S nodes
    for s_idx, s in enumerate(S_nodes, start=1):
        dot.append(
            f'    {s} [label=<S<sub>{s_idx}</sub>>, fillcolor="{col_S}", width=1.1, fixedsize=true];'
        )

    # K nodes
    for k_idx, k in enumerate(K_nodes, start=1):
        dot.append(
            f'    {k} [label=<K<sub>{k_idx}</sub>>, fillcolor="{col_K}", width=1.1, fixedsize=true];'
        )

    # α edges: P -> S
    for (_, s) in kin_alpha_edges:
        s_idx = int(s.replace("S", ""))
        dot.append(
            f'    P -> {s} [color="{col_alpha}", label=<&#945;<sub>{s_idx}</sub>>];'
        )

    # β edges: S -> K (regulatory contribution β_{k,s})
    for (s, k) in kin_beta_edges:
        s_idx = int(s.replace("S", ""))
        k_idx = int(k.replace("K", ""))
        dot.append(
            f'    {s} -> {k} [color="{col_beta}", label=<&#946;<sub>{k_idx},{s_idx}</sub>>];'
        )

    # Per-kinase β0 + multiple PSites (TF-style)
    for k_idx, k in enumerate(K_nodes, start=1):
        # Protein component (β0,k)
        k0 = f"{k}_0"
        dot.append(
            f'    {k0} [label=<K<sub>{k_idx}</sub>(t)>, fillcolor="{col_inp}", width=1.35, fixedsize=true];'
        )
        dot.append(
            f'    {k0} -> {k} [color="{col_beta}", label=<&#946;<sub>0,{k_idx}</sub>>];'
        )

        # PSite components (β_{p,k})
        for p in range(1, kin_psites + 1):
            kp = f"{k}_p{p}"
            dot.append(
                f'    {kp} [label=<PSite<sub>{p},{k_idx}</sub>(t)>, fillcolor="{col_inp}", width=1.55, fixedsize=true];'
            )
            dot.append(
                f'    {kp} -> {k} [color="{col_beta}", label=<&#946;<sub>{p},{k_idx}</sub>>];'
            )

    dot.append('  }')  # end cluster_kinopt

    # -------------------------
    # TFopt module
    # -------------------------
    dot.append('  subgraph cluster_tfopt {')
    dot.append('    label=""; color="white"; penwidth=0;')

    for j, tf in enumerate(TFs, start=1):
        dot.append(
            f'    {tf} [label=<TF<sub>{j}</sub>>, fillcolor="{col_S}", width=1.1, fixedsize=true];'
        )

        # Protein component input (β0,j)
        tf0 = f"{tf}_0"
        dot.append(
            f'    {tf0} [label=<TF<sub>{j}</sub>(t)>, fillcolor="{col_inp}", width=1.35, fixedsize=true];'
        )
        dot.append(
            f'    {tf0} -> {tf} [color="{col_beta}", label=<&#946;<sub>0,{j}</sub>>];'
        )

        # PSites (β_{p,j})
        for p in range(1, tf_psites + 1):
            ps = f"{tf}_p{p}"
            dot.append(
                f'    {ps} [label=<PSite<sub>{p},{j}</sub>(t)>, fillcolor="{col_inp}", width=1.55, fixedsize=true];'
            )
            dot.append(
                f'    {ps} -> {tf} [color="{col_beta}", label=<&#946;<sub>{p},{j}</sub>>];'
            )

        # α edge TF -> P (α_{i,j})
        dot.append(
            f'    {tf} -> P [color="{col_alpha}", label=<&#945;<sub>i,{j}</sub>>];'
        )

    dot.append('  }')  # end cluster_tfopt

    dot.append("}")
    dot_str = "\n".join(dot)

    # ---- Render (pygraphviz preferred) ----
    try:
        A = pgv.AGraph(string=dot_str)
        A.layout(prog=engine)
        A.draw(outfile, format=fmt)
    except Exception:
        try:
            graphs = pydot.graph_from_dot_data(dot_str)
            if not graphs:
                raise RuntimeError("pydot failed to parse DOT.")
            g = graphs[0]
            write_fn = getattr(g, f"write_{fmt}", None)
            if write_fn is None:
                g.write_raw(outfile)
            else:
                write_fn(outfile, prog=engine)
        except Exception as e:
            raise RuntimeError(
                "Could not render via pygraphviz or pydot. "
                "You likely need Graphviz + one of: pygraphviz, pydot.\n"
                "Install examples:\n"
                "  sudo apt-get install graphviz graphviz-dev\n"
                "  pip install pygraphviz\n"
                "or:\n"
                "  pip install pydot\n"
                f"Original error: {e}"
            )

    print(f"Saved: {outfile}")

if __name__ == "__main__":
    make_tfopt_diagram_dot("tfopt_diagram.png", engine="dot", dpi=300)
    make_kinopt_diagram_dot("kinopt_diagram.png", engine="dot", dpi=300)
    make_global_diagram_dot("global_diagram.png", engine="dot", dpi=300)