#!/usr/bin/env python3
"""
Export per-shared-node subnetworks in the original layouts of:
- input2: rows with columns like [GeneID, Psite, Kinase, ...] where Kinase is a set-like string
- input4: edges with columns like [Source, Target, ...] (all columns preserved)

For each shared node (appears in both networks), the script builds a merged graph and extracts
a k-hop neighborhood (or the full weakly connected component if --hops auto), then writes:

  <outdir>/<SHARED_NODE>/<SHARED_NODE>_input2.csv
  <outdir>/<SHARED_NODE>/<SHARED_NODE>_input4.csv

Also writes:
  <outdir>/INDEX.csv  (includes per-node counts and max hop distance reachable)

Optionally writes:
  <outdir>.zip  (zip of the outdir contents)

Usage: python export_subnetworks.py --hops 2 --outdir <outdir> <input2> <input4>

License: BSD-3-Clause
Author: Abhinav Mishra
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


def parse_setlike(value) -> List[str]:
    """
    Parse strings like:
      "{A, B, C}" or "A, B, C" or "['A','B']" (best-effort) into a list of tokens.
    Keeps tokens as-is except trimming quotes/whitespace.

    Args:
        value: The raw string or object from the DataFrame cell.

    Returns:
        List[str]: A list of cleaned string tokens.
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    # Common case: "{...}" indicating a set in string format
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    # Split on commas; tolerate extra spaces
    parts = [p.strip() for p in s.split(",") if p.strip()]

    # Strip surrounding quotes to handle JSON-like or Python-list-like strings
    cleaned = [re.sub(r"^['\"]|['\"]$", "", p) for p in parts]
    # Drop empties, preserve order but remove duplicates
    out = []
    seen = set()
    for x in cleaned:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def setlike_to_str(items: Iterable[str], sort_items: bool = True) -> str:
    """
    Converts an iterable of items back into a set-like string format "{A, B, C}".

    Args:
        items (Iterable[str]): The items to format.
        sort_items (bool): Whether to alphabetically sort the items in the output string.

    Returns:
        str: The formatted string.
    """
    items = [str(x) for x in items]
    # unique preserve first, then sort for stability (optional)
    seen = set()
    uniq = []
    for x in items:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    if sort_items:
        uniq = sorted(uniq)
    return "{%s}" % (", ".join(uniq))


def build_merged_graph(
        df2: pd.DataFrame,
        df4: pd.DataFrame,
        gene_col: str = "GeneID",
        psite_col: str = "Psite",
        kinase_col: str = "Kinase",
        source_col: str = "Source",
        target_col: str = "Target",
) -> nx.DiGraph:
    """
    Constructs a merged directed graph from the two input DataFrames.

    - Input2 (Kinase-Substrate): Treated as edges from Kinase -> Gene.
      Since the 'Kinase' column can contain multiple kinases (set-like),
      multiple edges are created for a single row.
    - Input4 (Regulatory/Interaction): Treated as edges from Source -> Target.

    Args:
        df2 (pd.DataFrame): The kinase-substrate dataframe.
        df4 (pd.DataFrame): The regulatory/interaction dataframe.
        gene_col (str): Column name for the target gene in df2.
        psite_col (str): Column name for the phosphosite in df2 (added as edge attribute).
        kinase_col (str): Column name for the kinase(s) in df2.
        source_col (str): Column name for the source node in df4.
        target_col (str): Column name for the target node in df4.

    Returns:
        nx.DiGraph: A NetworkX directed graph containing all nodes and edges.
    """
    G = nx.DiGraph()

    # input2 edges: Kinase -> GeneID
    for _, row in df2.iterrows():
        gene = str(row[gene_col])
        psite = str(row[psite_col])
        # Expand the set-like kinase column into individual edges
        kinases = parse_setlike(row[kinase_col])
        for k in kinases:
            G.add_edge(str(k), gene, edge_type="kinase", psite=psite, source="input2")

    # input4 edges: Source -> Target
    for s, t in df4[[source_col, target_col]].astype(str).itertuples(index=False):
        G.add_edge(s, t, edge_type="reg", source="input4")

    return G


def save_subnetwork_png(
        G_merged: nx.DiGraph,
        center: str,
        nodes: Set[str],
        out_png: Path,
        shared_nodes: Optional[Set[str]] = None,
        label_mode: str = "all",  # "all" | "smart" | "center_only"
        dpi: int = 300,
) -> None:
    """
    Save a PNG visualization of the extracted subnetwork.

    Styling:
      - solid edges: edge_type == "reg" (input4)
      - dashed edges: edge_type == "kinase" (input2)
      - center node emphasized (large size)
      - nodes that are shared (if provided) are slightly larger than others

    Parameters
    ----------
    G_merged:
        The merged directed graph containing both networks with edge attributes:
          edge_type in {"reg","kinase"}.
    center:
        The shared node for this subnetwork (the focal point).
    nodes:
        The node set to visualize (e.g., k-hop neighborhood).
    out_png:
        Output file path.
    shared_nodes:
        Optional set of nodes that are shared between the two original networks.
        Used for visual distinction (sizing).
    label_mode:
        "all": label every node (can get messy)
        "center_only": only label the center
        "smart": label center + nodes with degree>=3 within the subgraph (default)
    dpi:
        Output resolution.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Create a subgraph containing only the relevant nodes
    subG = G_merged.subgraph(nodes).copy()
    if subG.number_of_nodes() == 0:
        raise ValueError("Subgraph is empty; nothing to plot.")

    # Layout (undirected for geometry stability - springs work better without direction constraints)
    pos = nx.spring_layout(subG.to_undirected(as_view=True), seed=7, k=0.9 / max(subG.number_of_nodes() ** 0.5, 1))

    # Node sizing logic
    shared_nodes = shared_nodes or set()
    node_sizes = []
    for n in subG.nodes():
        if n == center:
            node_sizes.append(900)  # Largest for the center
        elif n in shared_nodes:
            node_sizes.append(260)  # Medium for other shared nodes
        else:
            node_sizes.append(160)  # Small for peripheral nodes

    # Split edges by type for different styling
    reg_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("edge_type") == "reg"]
    kin_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("edge_type") == "kinase"]

    plt.figure(figsize=(18, 10))
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, linewidths=0.7)

    # Draw regular edges (solid)
    if reg_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=reg_edges, arrows=True, arrowsize=12, width=1.2)
    # Draw kinase edges (dashed) to visually distinguish interaction types
    if kin_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=kin_edges, arrows=True, arrowsize=12, width=1.4, style="dashed")

    # Labeling logic
    if label_mode == "center_only":
        labels = {center: center}
        font_size = 10
    elif label_mode == "all":
        labels = {n: n for n in subG.nodes()}
        font_size = 8
    else:
        # smart: reduce clutter by only labeling hubs and the center
        deg = dict(subG.degree())
        labels = {n: n for n in subG.nodes() if (n == center or deg.get(n, 0) >= 3)}
        font_size = 9

    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=font_size)

    plt.title(f"{center} subnetwork | nodes={subG.number_of_nodes()} edges={subG.number_of_edges()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def compute_network_nodes_input2(
        df2: pd.DataFrame, gene_col: str, kinase_col: str
) -> Set[str]:
    """Extract all unique nodes (Genes and Kinases) from input2."""
    nodes = set(df2[gene_col].astype(str))
    kin_nodes = set()
    for v in df2[kinase_col]:
        kin_nodes.update(parse_setlike(v))
    nodes |= set(map(str, kin_nodes))
    return nodes


def compute_network_nodes_input4(
        df4: pd.DataFrame, source_col: str, target_col: str
) -> Set[str]:
    """Extract all unique nodes (Sources and Targets) from input4."""
    return set(df4[source_col].astype(str)) | set(df4[target_col].astype(str))


def k_hop_nodes_undirected(G: nx.DiGraph, center: str, k: int) -> Set[str]:
    """
    Finds all nodes within k hops of the center in the undirected sense.

    k=0 -> {center}
    k=1 -> center + direct neighbors
    """
    if k < 0:
        return {center}
    # Convert to undirected to traverse both incoming and outgoing edges
    UG = G.to_undirected(as_view=True)
    visited = {center}
    q = deque([(center, 0)])

    # BFS traversal
    while q:
        node, dist = q.popleft()
        if dist == k:
            continue
        for nb in UG.neighbors(node):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, dist + 1))
    return visited


def weak_component_nodes(G: nx.DiGraph, center: str) -> Set[str]:
    """
    Extracts the full weakly connected component containing the center node.
    This includes all nodes reachable from the center by ignoring edge direction.
    """
    UG = G.to_undirected()
    if center not in UG:
        return {center}
    # networkx gives connected components for undirected graphs
    for comp in nx.connected_components(UG):
        if center in comp:
            return set(comp)
    return {center}


def max_hops_reachable_in_component(G: nx.DiGraph, center: str) -> int:
    """
    Calculates the maximum undirected shortest-path distance (eccentricity)
    from the center to any other node in its weak component.
    If isolated node -> 0.
    """
    UG = G.to_undirected()
    if center not in UG:
        return 0
    comp = weak_component_nodes(G, center)
    sub = UG.subgraph(comp)
    # Calculate shortest paths to all reachable nodes
    lengths = nx.single_source_shortest_path_length(sub, center)
    if not lengths:
        return 0
    return max(lengths.values())


def export_subnetworks(
        df2: pd.DataFrame,
        df4: pd.DataFrame,
        outdir: Path,
        hops: Optional[int],  # None means auto/full component
        gene_col: str,
        psite_col: str,
        kinase_col: str,
        source_col: str,
        target_col: str,
        sort_kinases_in_set: bool,
        make_zip: bool,
) -> Tuple[Path, Path]:
    """
    Main logic for generating subnetwork files.

    1. Builds a master merged graph.
    2. Identifies shared nodes (nodes present in both datasets).
    3. For each shared node:
       - Determines the set of relevant nodes (k-hop or full component).
       - Filters df4 (edges) to include only edges where both source and target are in the set.
       - Filters df2 (kinase-substrate) to include rows where the Gene is in the set
         AND updates the Kinase column to only include kinases present in the set.
       - Saves CSVs and generates a visualization.
    4. Creates an index CSV and optional ZIP archive.

    Returns:
        Tuple[Path, Path]: Paths to the index file and the zip file.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Build master graph
    G = build_merged_graph(
        df2, df4,
        gene_col=gene_col, psite_col=psite_col, kinase_col=kinase_col,
        source_col=source_col, target_col=target_col
    )

    # 2. Identify shared nodes
    nodes_net2 = compute_network_nodes_input2(df2, gene_col, kinase_col)
    nodes_net4 = compute_network_nodes_input4(df4, source_col, target_col)
    shared_nodes = sorted(nodes_net2 & nodes_net4)

    index_rows = []

    # 3. Iterate over each shared node to create subnetworks
    for center in shared_nodes:
        max_hops = max_hops_reachable_in_component(G, center)

        # Determine node set based on hop strategy
        if hops is None:
            # auto: full weak component
            nodes = weak_component_nodes(G, center)
            used_hops = max_hops
        else:
            nodes = k_hop_nodes_undirected(G, center, hops)
            used_hops = hops

        # --- Filter Input4 (Reg/Interaction) ---
        # Keep edges only if BOTH source and target are inside the subnetwork node set
        df4_sub = df4[
            df4[source_col].astype(str).isin(nodes) & df4[target_col].astype(str).isin(nodes)
            ].copy()

        # --- Filter Input2 (Kinase-Substrate) ---
        # This is more complex because 'Kinase' column is a set-like string.
        # We must filter the row AND modify the content of the Kinase column.
        rows = []
        for _, row in df2.iterrows():
            gene = str(row[gene_col])
            # The target gene must be in the network
            if gene not in nodes:
                continue

            # Parse kinases, keep only those that are in the network
            kinases_in_nodes = [str(k) for k in parse_setlike(row[kinase_col]) if str(k) in nodes]

            # If no kinases remain for this site in this subnetwork, skip the row
            if not kinases_in_nodes:
                continue

            new_row = row.copy()
            # Re-serialize the valid kinases back into the string format
            new_row[kinase_col] = setlike_to_str(
                kinases_in_nodes, sort_items=sort_kinases_in_set
            )
            rows.append(new_row)

        df2_sub = pd.DataFrame(rows, columns=df2.columns)

        # Save results
        node_dir = outdir / center
        node_dir.mkdir(exist_ok=True)

        f2 = node_dir / f"{center}_input2.csv"
        f4 = node_dir / f"{center}_input4.csv"
        df2_sub.to_csv(f2, index=False)
        df4_sub.to_csv(f4, index=False)

        # Record metadata for INDEX.csv
        index_rows.append(
            {
                "shared_node": center,
                "requested_hops": used_hops,
                "max_hops_reachable": max_hops,
                "subnetwork_nodes": len(nodes),
                "input2_rows": len(df2_sub),
                "input4_rows": len(df4_sub),
                "relpath_input2": str(f2.relative_to(outdir)),
                "relpath_input4": str(f4.relative_to(outdir)),
            }
        )

        png_path = (outdir / center / f"{center}_graph.png")

        save_subnetwork_png(
            G_merged=G,
            center=center,
            nodes=nodes,
            out_png=png_path,
            shared_nodes=set(shared_nodes),
            label_mode="all",
            dpi=300,
        )

    # 4. Finalize Output
    index_df = pd.DataFrame(index_rows)
    index_path = outdir / "INDEX.csv"
    index_df.to_csv(index_path, index=False)

    zip_path = outdir.with_suffix(".zip")
    if make_zip:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            # Write index first
            z.write(index_path, arcname="INDEX.csv")
            # Then every other CSV under outdir
            for p in sorted(outdir.rglob("*.csv")):
                if p.name == "INDEX.csv":
                    continue
                z.write(p, arcname=str(p.relative_to(outdir)))

    return index_path, zip_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export per-shared-node subnetworks in the original input2/input4 layouts."
    )
    ap.add_argument("--input2", required=True, help="Path to input2.csv (Kinase/GeneID table).")
    ap.add_argument("--input4", required=True, help="Path to input4.csv (Source/Target edge list).")
    ap.add_argument("--outdir", required=True, help="Output directory for subnetworks.")
    ap.add_argument(
        "--hops",
        default="1",
        help="Neighborhood hops (undirected). Use an integer (e.g., 1,2,3) or 'auto' for full component.",
    )
    ap.add_argument(
        "--zip",
        action="store_true",
        help="Also create <outdir>.zip containing INDEX.csv and all subnetwork CSVs.",
    )

    # Column names (override if your files differ)
    ap.add_argument("--gene-col", default="GeneID")
    ap.add_argument("--psite-col", default="Psite")
    ap.add_argument("--kinase-col", default="Kinase")
    ap.add_argument("--source-col", default="Source")
    ap.add_argument("--target-col", default="Target")

    ap.add_argument(
        "--no-sort-kinases",
        action="store_true",
        help="Do not sort kinase set strings; keep first-seen order.",
    )

    args = ap.parse_args()

    input2_path = Path(args.input2)
    input4_path = Path(args.input4)
    outdir = Path(args.outdir)

    # Input validation
    if not input2_path.exists():
        print(f"ERROR: input2 not found: {input2_path}", file=sys.stderr)
        return 2
    if not input4_path.exists():
        print(f"ERROR: input4 not found: {input4_path}", file=sys.stderr)
        return 2

    df2 = pd.read_csv(input2_path)
    df4 = pd.read_csv(input4_path)

    # Column validation
    required2 = {args.gene_col, args.psite_col, args.kinase_col}
    required4 = {args.source_col, args.target_col}
    if not required2.issubset(df2.columns):
        print(f"ERROR: input2 missing columns: {sorted(required2 - set(df2.columns))}", file=sys.stderr)
        return 2
    if not required4.issubset(df4.columns):
        print(f"ERROR: input4 missing columns: {sorted(required4 - set(df4.columns))}", file=sys.stderr)
        return 2

    # Parse hops argument
    if str(args.hops).lower() == "auto":
        hops = None
    else:
        try:
            hops = int(args.hops)
            if hops < 0:
                raise ValueError
        except ValueError:
            print("ERROR: --hops must be a non-negative integer or 'auto'.", file=sys.stderr)
            return 2

    index_path, zip_path = export_subnetworks(
        df2=df2,
        df4=df4,
        outdir=outdir,
        hops=hops,
        gene_col=args.gene_col,
        psite_col=args.psite_col,
        kinase_col=args.kinase_col,
        source_col=args.source_col,
        target_col=args.target_col,
        sort_kinases_in_set=not args.no_sort_kinases,
        make_zip=bool(args.zip),
    )

    print(f"Wrote: {index_path}")
    if args.zip:
        print(f"Wrote: {zip_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
