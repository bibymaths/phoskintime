# lossfn.py
from numba import njit
from phoskintime_global.config import MODEL

EPS = 1e-9

@njit(fastmath=True, cache=True, nogil=True)
def loss_function_noncomb(Y,
                          p_prot, t_prot, obs_prot, w_prot,
                          p_rna,  t_rna,  obs_rna,  w_rna,
                          p_pho,  t_pho,  obs_pho,  w_pho,
                          prot_map, rna_base_idx):
    # ---- protein total ----
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        tot_t = Y[t_idx, start + 1]
        tot_0 = Y[0,     start + 1]
        for s in range(n_sites):
            tot_t += Y[t_idx, start + 2 + s]
            tot_0 += Y[0,     start + 2 + s]

        pred_fc = (tot_t if tot_t > EPS else EPS) / (tot_0 if tot_0 > EPS else EPS)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * diff * diff

    # ---- RNA ----
    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]
        pred_fc = (R_t if R_t > EPS else EPS) / (R_b if R_b > EPS else EPS)

        diff = obs_rna[k] - pred_fc
        loss_r += w_rna[k] * diff * diff

    # ---- phospho protein (exclude P0) ----
    loss_ph = 0.0
    for k in range(p_pho.size):
        p_idx = p_pho[k]
        t_idx = t_pho[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        ph_t = 0.0
        ph_0 = 0.0
        for s in range(n_sites):
            ph_t += Y[t_idx, start + 2 + s]
            ph_0 += Y[0,     start + 2 + s]

        pred_fc = (ph_t if ph_t > EPS else EPS) / (ph_0 if ph_0 > EPS else EPS)
        diff = obs_pho[k] - pred_fc
        loss_ph += w_pho[k] * diff * diff

    return loss_p, loss_r, loss_ph


@njit(fastmath=True, cache=True, nogil=True)
def loss_function_comb(Y,
                       p_prot, t_prot, obs_prot, w_prot,
                       p_rna,  t_rna,  obs_rna,  w_rna,
                       p_pho,  t_pho,  obs_pho,  w_pho,
                       prot_map, rna_base_idx):
    # ---- protein total ----
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1  # mask=0

        tot_t = 0.0
        tot_0 = 0.0
        for m in range(nstates):
            tot_t += Y[t_idx, p0 + m]
            tot_0 += Y[0,     p0 + m]

        pred_fc = (tot_t if tot_t > EPS else EPS) / (tot_0 if tot_0 > EPS else EPS)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * diff * diff

    # ---- RNA ----
    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]
        pred_fc = (R_t if R_t > EPS else EPS) / (R_b if R_b > EPS else EPS)

        diff = obs_rna[k] - pred_fc
        loss_r += w_rna[k] * diff * diff

    # ---- phospho protein = total - unphospho(mask=0) ----
    loss_ph = 0.0
    for k in range(p_pho.size):
        p_idx = p_pho[k]
        t_idx = t_pho[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1

        tot_t = 0.0
        tot_0 = 0.0
        for m in range(nstates):
            tot_t += Y[t_idx, p0 + m]
            tot_0 += Y[0,     p0 + m]

        ph_t = tot_t - Y[t_idx, p0]
        ph_0 = tot_0 - Y[0,     p0]

        pred_fc = (ph_t if ph_t > EPS else EPS) / (ph_0 if ph_0 > EPS else EPS)
        diff = obs_pho[k] - pred_fc
        loss_ph += w_pho[k] * diff * diff

    return loss_p, loss_r, loss_ph


LOSS_FN = loss_function_comb if MODEL == 2 else loss_function_noncomb