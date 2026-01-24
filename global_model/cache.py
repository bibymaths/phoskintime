"""
Fast Loss Data Preparation Module.

This script handles the crucial pre-processing step for the optimization loop.
Instead of performing slow dictionary lookups and string comparisons inside the
loss function (which runs thousands of times), this module maps all experimental
data (RNA, Protein, Phospho) onto integer-based grid indices *once*.

The output is a dictionary of contiguous NumPy arrays (indices, observations, weights)
that can be passed directly to a fast JIT-compiled or Cython loss function.
"""

import numpy as np
import pandas as pd

from global_model.config import MODEL


def prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, time_grid):
    """
    Converts pandas DataFrames of experimental data into integer-based arrays
    aligned with the simulation state vector and time grid.

    This function performs "pre-indexing":
    1.  Maps float timepoints to integer indices in `time_grid`.
    2.  Maps string protein names to integer state indices (`p2i`).
    3.  Maps specific phosphosites (e.g., "S473") to relative state offsets.
    4.  Constructs a `prot_map` for fast state slicing.

    Args:
        idx (IndexMap): Object containing mappings (p2i, sites, n_sites, etc.).
        df_prot (pd.DataFrame): Protein data columns [protein, time, fc, w].
        df_rna (pd.DataFrame): RNA data columns [protein, time, fc, w].
        df_pho (pd.DataFrame): Phospho data columns [protein, psite, time, fc, w].
        time_grid (array-like): The fixed time points of the simulation solver.

    Returns:
        dict: A dictionary containing keyed NumPy arrays (e.g., 'p_prot', 'obs_prot')
              ready for the fast loss function.
    """

    # Pre-compute time index map: Time Value (float) -> Grid Index (int)
    t_map = {float(t): i for i, t in enumerate(np.asarray(time_grid, dtype=float))}

    def _map_times(t_arr):
        """Helper to vectorize mapping of experimental times to grid indices."""
        t_arr = np.asarray(t_arr, dtype=float)
        out = np.empty(t_arr.shape[0], dtype=np.int32)
        for i, t in enumerate(t_arr):
            if t not in t_map:
                raise ValueError(
                    f"Time {t} not found in time_grid. "
                    f"Fix by rounding/normalizing times or passing the correct grid."
                )
            out[i] = t_map[t]
        return out

    def get_indices_basic(df, p2i_map):
        """
        Processes 'basic' data (Protein or RNA) where the entity is identified
        only by protein name (no site specificity).
        """
        # Map protein names to integer indices
        prots = df["protein"].values
        p_idxs = np.empty(len(prots), dtype=np.int32)
        for i, p in enumerate(prots):
            if p not in p2i_map:
                raise ValueError(f"Protein '{p}' not in idx.p2i (interaction network index).")
            p_idxs[i] = p2i_map[p]

        # Map times
        t_idxs = _map_times(df["time"].values)

        # Extract observations + weights
        obs = np.ascontiguousarray(df["fc"].values, dtype=np.float64)
        if "w" in df.columns:
            ws = np.ascontiguousarray(df["w"].values, dtype=np.float64)
        else:
            ws = np.ones(len(df), dtype=np.float64)

        return (np.ascontiguousarray(p_idxs, dtype=np.int32),
                np.ascontiguousarray(t_idxs, dtype=np.int32),
                obs,
                ws)

    # --- Phospho data needs site-specific indices ---
    # Build a lookup for every protein: Site Name -> Local Index (0..n_sites-1)
    site_maps = []
    for i in range(idx.N):
        mp = {s: j for j, s in enumerate(idx.sites[i])}
        site_maps.append(mp)

    def get_indices_phospho(df):
        """
        Processes phosphorylation data. Maps (Protein, P-Site) -> (Protein Index, Site Index).
        """
        p_idxs = []
        s_idxs = []
        t_idxs = []
        obs = []
        ws = []

        for _, row in df.iterrows():
            p = row["protein"]
            if p not in idx.p2i:
                continue
            pi = idx.p2i[p]

            # Look up the specific site index for this protein
            s = row["psite"]
            if s not in site_maps[pi]:
                continue  # Site exists in data but not in model structure; ignore.

            p_idxs.append(pi)
            s_idxs.append(site_maps[pi][s])

            t = float(row["time"])
            if t not in t_map:
                raise ValueError(f"Time {t} not found in time_grid")
            t_idxs.append(t_map[t])

            obs.append(float(row["fc"]))
            ws.append(float(row["w"]) if "w" in row and pd.notna(row["w"]) else 1.0)

        return (
            np.asarray(p_idxs, dtype=np.int32),
            np.asarray(s_idxs, dtype=np.int32),
            np.asarray(t_idxs, dtype=np.int32),
            np.asarray(obs, dtype=np.float64),
            np.asarray(ws, dtype=np.float64),
        )

    # Process all three data types
    p_prot, t_prot, obs_prot, w_prot = get_indices_basic(df_prot, idx.p2i)
    p_rna, t_rna, obs_rna, w_rna = get_indices_basic(df_rna, idx.p2i)
    p_pho, s_pho, t_pho, obs_pho, w_pho = get_indices_phospho(df_pho)

    # prot_map: A lookup table for the loss function to know where a protein's data starts in Y.
    # Structure: [Start Index in Y, Count (sites or states)]
    prot_map = np.zeros((idx.N, 2), dtype=np.int32)
    for i in range(idx.N):
        sl = idx.block(i)
        prot_map[i, 0] = sl.start
        # If MODEL==2, the state vector structure might differ, requiring total states vs just site count.
        prot_map[i, 1] = int(idx.n_states[i]) if MODEL == 2 else int(idx.n_sites[i])

    return {
        "p_prot": p_prot, "t_prot": t_prot, "obs_prot": obs_prot, "w_prot": w_prot,
        "p_rna": p_rna, "t_rna": t_rna, "obs_rna": obs_rna, "w_rna": w_rna,
        "p_pho": p_pho, "s_pho": s_pho, "t_pho": t_pho, "obs_pho": obs_pho, "w_pho": w_pho,
        "prot_map": np.ascontiguousarray(prot_map, dtype=np.int32),
        "n_p": max(1, len(obs_prot)),
        "n_r": max(1, len(obs_rna)),
        "n_ph": max(1, len(obs_pho)),
    }