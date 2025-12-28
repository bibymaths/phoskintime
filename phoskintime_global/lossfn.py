from numba import njit
from phoskintime_global.config import MODEL

@njit(fastmath=True, cache=True, nogil=True)
def loss_function_noncomb(Y, p_prot, t_prot, obs_prot, w_prot,
                          p_rna, t_rna, obs_rna, w_rna,
                          prot_map, rna_base_idx):
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        # total protein at t
        tot_t = Y[t_idx, start + 1]
        for s in range(n_sites):
            tot_t += Y[t_idx, start + 2 + s]

        # total protein at 0
        tot_0 = Y[0, start + 1]
        for s in range(n_sites):
            tot_0 += Y[0, start + 2 + s]

        pred_fc = (tot_t if tot_t > 1e-9 else 1e-9) / (tot_0 if tot_0 > 1e-9 else 1e-9)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * diff * diff

    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]
        pred_fc = (R_t if R_t > 1e-9 else 1e-9) / (R_b if R_b > 1e-9 else 1e-9)

        diff = obs_rna[k] - pred_fc
        loss_r += w_rna[k] * diff * diff

    return loss_p, loss_r


@njit(fastmath=True, cache=True, nogil=True)
def loss_function_comb(Y, p_prot, t_prot, obs_prot, w_prot,
                       p_rna, t_rna, obs_rna, w_rna,
                       prot_map, rna_base_idx):
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1

        tot_t = 0.0
        tot_0 = 0.0
        for m in range(nstates):
            tot_t += Y[t_idx, p0 + m]
            tot_0 += Y[0, p0 + m]

        pred_fc = (tot_t if tot_t > 1e-9 else 1e-9) / (tot_0 if tot_0 > 1e-9 else 1e-9)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * diff * diff

    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]
        pred_fc = (R_t if R_t > 1e-9 else 1e-9) / (R_b if R_b > 1e-9 else 1e-9)

        diff = obs_rna[k] - pred_fc
        loss_r += w_rna[k] * diff * diff

    return loss_p, loss_r

LOSS_FN = loss_function_comb if MODEL == 2 else loss_function_noncomb
