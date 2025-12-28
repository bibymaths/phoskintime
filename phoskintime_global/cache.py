import numpy as np

from phoskintime_global.config import MODEL


def prepare_fast_loss_data(idx, df_prot, df_rna, time_grid):
    t_map = {t: i for i, t in enumerate(time_grid)}

    def get_indices(df, p2i_map):
        p_idxs = np.array([p2i_map[p] for p in df["protein"].values], dtype=np.int32)
        t_idxs = np.array([t_map[t] for t in df["time"].values], dtype=np.int32)
        obs = df["fc"].values.astype(np.float64)
        ws = df["w"].values.astype(np.float64)
        return p_idxs, t_idxs, obs, ws

    p_prot, t_prot, obs_prot, w_prot = get_indices(df_prot, idx.p2i)
    p_rna, t_rna, obs_rna, w_rna = get_indices(df_rna, idx.p2i)

    prot_map = np.zeros((idx.N, 2), dtype=np.int32)
    for i in range(idx.N):
        sl = idx.block(i)
        prot_map[i, 0] = sl.start
        if MODEL == 2:
            prot_map[i, 1] = idx.n_states[i]
        else:
            prot_map[i, 1] = idx.n_sites[i]

    return {
        "p_prot": p_prot, "t_prot": t_prot, "obs_prot": obs_prot, "w_prot": w_prot,
        "p_rna": p_rna, "t_rna": t_rna, "obs_rna": obs_rna, "w_rna": w_rna,
        "prot_map": prot_map,
        "n_p": max(1, len(obs_prot)),
        "n_r": max(1, len(obs_rna))
    }
