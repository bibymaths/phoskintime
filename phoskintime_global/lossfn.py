# lossfn.py
from numba import njit
from phoskintime_global.config import MODEL

EPS = 1e-9

@njit(fastmath=True, cache=True, nogil=True)
def loss_function_noncomb(
    Y,
    p_prot, t_prot, obs_prot, w_prot,
    p_rna,  t_rna,  obs_rna,  w_rna,
    p_pho, s_pho, t_pho,  obs_pho,  w_pho,
    prot_map,
    prot_base_idx, rna_base_idx, pho_base_idx
):
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        tot_t = Y[t_idx, start + 1]
        tot_b = Y[prot_base_idx, start + 1]
        for s in range(n_sites):
            tot_t += Y[t_idx, start + 2 + s]
            tot_b += Y[prot_base_idx, start + 2 + s]

        pred_fc = (tot_t if tot_t > EPS else EPS) / (tot_b if tot_b > EPS else EPS)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * diff * diff

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

    loss_ph = 0.0
    for k in range(p_pho.size):
        p_idx = p_pho[k]
        s_idx = s_pho[k]
        t_idx = t_pho[k]
        start = prot_map[p_idx, 0]

        ph_t = Y[t_idx,      start + 2 + s_idx]
        ph_b = Y[pho_base_idx, start + 2 + s_idx]

        pred_fc = (ph_t if ph_t > EPS else EPS) / (ph_b if ph_b > EPS else EPS)
        diff = obs_pho[k] - pred_fc
        loss_ph += w_pho[k] * diff * diff

    return loss_p, loss_r, loss_ph


@njit(fastmath=True, cache=True, nogil=True)
def loss_function_comb(
    Y,
    p_prot, t_prot, obs_prot, w_prot,
    p_rna,  t_rna,  obs_rna,  w_rna,
    p_pho, s_pho, t_pho,  obs_pho,  w_pho,
    prot_map,
    prot_base_idx, rna_base_idx, pho_base_idx
):
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1

        tot_t = 0.0
        tot_b = 0.0
        for m in range(nstates):
            tot_t += Y[t_idx, p0 + m]
            tot_b += Y[prot_base_idx, p0 + m]

        pred_fc = (tot_t if tot_t > EPS else EPS) / (tot_b if tot_b > EPS else EPS)
        diff = obs_prot[k] - pred_fc
        loss_p += w_prot[k] * diff * diff

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

    loss_ph = 0.0
    for k in range(p_pho.size):
        p_idx = p_pho[k]
        j = s_pho[k]               # site index
        t_idx = t_pho[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1

        ph_t = 0.0
        ph_b = 0.0
        for m in range(nstates):
            if (m >> j) & 1:
                ph_t += Y[t_idx,       p0 + m]
                ph_b += Y[pho_base_idx, p0 + m]

        pred_fc = (ph_t if ph_t > EPS else EPS) / (ph_b if ph_b > EPS else EPS)
        diff = obs_pho[k] - pred_fc
        loss_ph += w_pho[k] * diff * diff

    return loss_p, loss_r, loss_ph

LOSS_FN = loss_function_comb if MODEL == 2 else loss_function_noncomb