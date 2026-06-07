"""Convert observation data frames into compact numeric arrays for fast loss evaluation; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config."""

import numpy as np
import pandas as pd

from networkmodel.config import MODEL


def prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, time_grid):
    """Prepare numeric loss arrays from observation data frames
    
    Args:
        idx: Input value used by this routine.
        df_prot: Input value used by this routine.
        df_rna: Input value used by this routine.
        df_pho: Input value used by this routine.
        time_grid: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
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

    def _empty_basic():
        return (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))

    def _empty_phospho():
        return (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.float64))

    # Process only available data types. Empty/missing layers remain empty arrays
    # and are skipped by the scalar JAX objective rather than padded with zeros.
    p_prot, t_prot, obs_prot, w_prot = _empty_basic() if df_prot is None or df_prot.empty else get_indices_basic(
        df_prot, idx.p2i)
    p_rna, t_rna, obs_rna, w_rna = _empty_basic() if df_rna is None or df_rna.empty else get_indices_basic(df_rna,
                                                                                                           idx.p2i)
    p_pho, s_pho, t_pho, obs_pho, w_pho = _empty_phospho() if df_pho is None or df_pho.empty else get_indices_phospho(
        df_pho)

    # prot_map: A lookup table for the global JAX loss to know where each
    # protein's block starts in the flattened ODE state vector. The global
    # networkmodel state layout is:
    #   MODEL 0/1: [mRNA, unphosphorylated protein, phospho_site_0, ...]
    #   MODEL 2:   [mRNA, combinatorial protein_state_0, ... protein_state_(2^n-1)]
    # The second column is therefore n_sites for MODEL 0/1 and n_states for
    # MODEL 2. n_sites is also exported separately so MODEL 2 phospho rows can
    # aggregate states with the requested site bit set.
    prot_map = np.zeros((idx.N, 2), dtype=np.int32)
    for i in range(idx.N):
        sl = idx.block(i)
        prot_map[i, 0] = sl.start
        prot_map[i, 1] = int(idx.n_states[i]) if MODEL == 2 else int(idx.n_sites[i])

    return {
        "p_prot": p_prot, "t_prot": t_prot, "obs_prot": obs_prot, "w_prot": w_prot,
        "p_rna": p_rna, "t_rna": t_rna, "obs_rna": obs_rna, "w_rna": w_rna,
        "p_pho": p_pho, "s_pho": s_pho, "t_pho": t_pho, "obs_pho": obs_pho, "w_pho": w_pho,
        "prot_map": np.ascontiguousarray(prot_map, dtype=np.int32),
        "n_sites": np.ascontiguousarray(idx.n_sites, dtype=np.int32),
        "state_layout": "combinatorial" if MODEL == 2 else "standard",
        "backend_mode": "networkmodel",
        "n_p": len(obs_prot),
        "n_r": len(obs_rna),
        "n_ph": len(obs_pho),
    }
