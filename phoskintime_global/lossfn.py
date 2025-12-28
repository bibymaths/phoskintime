from numba import njit


@njit(fastmath=True, cache=True, nogil=True)
def jit_loss_core(Y, p_prot, t_prot, obs_prot, w_prot,
                  p_rna, t_rna, obs_rna, w_rna,
                  prot_map, rna_base_idx):
    loss_p = 0.0
    for k in range(len(p_prot)):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        P_t = Y[t_idx, start + 1]
        Ps_t = 0.0
        for s in range(n_sites):
            Ps_t += Y[t_idx, start + 2 + s]
        tot_t = P_t + Ps_t

        P_0 = Y[0, start + 1]
        Ps_0 = 0.0
        for s in range(n_sites):
            Ps_0 += Y[0, start + 2 + s]
        tot_0 = P_0 + Ps_0

        pred_fc = max(tot_t, 1e-9) / max(tot_0, 1e-9)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * (diff * diff)

    loss_r = 0.0
    for k in range(len(p_rna)):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]
        pred_fc = max(R_t, 1e-9) / max(R_b, 1e-9)

        diff = obs_rna[k] - pred_fc
        loss_r += w_rna[k] * (diff * diff)

    return loss_p, loss_r
