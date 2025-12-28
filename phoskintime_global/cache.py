import numpy as np
import pandas as pd

from phoskintime_global.config import MODEL


def prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, time_grid):
    t_map = {float(t): i for i, t in enumerate(np.asarray(time_grid, dtype=float))}

    def _map_times(t_arr):
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
        # protein indices
        prots = df["protein"].values
        p_idxs = np.empty(len(prots), dtype=np.int32)
        for i, p in enumerate(prots):
            if p not in p2i_map:
                raise ValueError(f"Protein '{p}' not in idx.p2i (interaction network index).")
            p_idxs[i] = p2i_map[p]

        # time indices
        t_idxs = _map_times(df["time"].values)

        # obs + weights
        obs = np.ascontiguousarray(df["fc"].values, dtype=np.float64)
        if "w" in df.columns:
            ws = np.ascontiguousarray(df["w"].values, dtype=np.float64)
        else:
            ws = np.ones(len(df), dtype=np.float64)

        return (np.ascontiguousarray(p_idxs, dtype=np.int32),
                np.ascontiguousarray(t_idxs, dtype=np.int32),
                obs,
                ws)

    # --- phospho needs site indices ---
    # build per-protein dict: psite -> site_index
    site_maps = []
    for i in range(idx.N):
        mp = {s: j for j, s in enumerate(idx.sites[i])}
        site_maps.append(mp)

    def get_indices_phospho(df):
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

            s = row["psite"]
            if s not in site_maps[pi]:
                continue  # truly drop it

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

    p_prot, t_prot, obs_prot, w_prot = get_indices_basic(df_prot, idx.p2i)
    p_rna,  t_rna,  obs_rna,  w_rna  = get_indices_basic(df_rna,  idx.p2i)
    p_pho,  s_pho,  t_pho,  obs_pho,  w_pho  = get_indices_phospho(df_pho)

    # prot_map: start offset and "count" (depends on MODEL)
    # For MODEL!=2: your state layout is [R, P, P1..Pns], so totals depend on how loss uses this.
    prot_map = np.zeros((idx.N, 2), dtype=np.int32)
    for i in range(idx.N):
        sl = idx.block(i)
        prot_map[i, 0] = sl.start
        prot_map[i, 1] = int(idx.n_states[i]) if MODEL == 2 else int(idx.n_sites[i])

    return {
        "p_prot": p_prot, "t_prot": t_prot, "obs_prot": obs_prot, "w_prot": w_prot,
        "p_rna":  p_rna,  "t_rna":  t_rna,  "obs_rna":  obs_rna,  "w_rna":  w_rna,
        "p_pho":  p_pho, "s_pho": s_pho, "t_pho": t_pho, "obs_pho": obs_pho, "w_pho": w_pho,
        "prot_map": np.ascontiguousarray(prot_map, dtype=np.int32),
        "n_p":  max(1, len(obs_prot)),
        "n_r":  max(1, len(obs_rna)),
        "n_ph": max(1, len(obs_pho)),
    }

