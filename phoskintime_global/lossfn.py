# lossfn.py
import numpy as np
from numba import njit
from phoskintime_global.config import MODEL, LOSS_MODE

EPS = 1e-9

@njit(fastmath=True, cache=True, nogil=True)
def sq(diff):
    return diff * diff

@njit(fastmath=True, cache=True, nogil=True)
def huber(diff, delta=1.0):
    a = diff if diff >= 0.0 else -diff
    if a <= delta:
        return 0.5 * diff * diff
    return delta * (a - 0.5 * delta)

@njit(fastmath=True, cache=True, nogil=True)
def pseudo_huber(diff, delta=1.0):
    x = diff / delta
    return (delta * delta) * ((1.0 + x * x) ** 0.5 - 1.0)

@njit(fastmath=True, cache=True, nogil=True)
def charbonnier(diff, eps=1e-3):
    return (diff * diff + eps * eps) ** 0.5 - eps

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

        if LOSS_MODE == 0:
            loss_p += w_prot[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_p += w_prot[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_prot[k] + EPS)
            loss_p += w_prot[k] * pseudo_huber(diff, 0.5)
        else:
            loss_p += w_prot[k] * charbonnier(diff, 1e-3)

    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]

        pred_fc = (R_t if R_t > EPS else EPS) / (R_b if R_b > EPS else EPS)

        diff = obs_rna[k] - pred_fc

        if LOSS_MODE == 0:
            loss_r += w_rna[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_r += w_rna[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_rna[k] + EPS)
            loss_r += w_rna[k] * pseudo_huber(diff, 0.5)
        else:
            loss_r += w_rna[k] * charbonnier(diff, 1e-3)

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

        if LOSS_MODE == 0:
            loss_ph += w_pho[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_ph += w_pho[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_pho[k] + EPS)
            loss_ph += w_pho[k] * pseudo_huber(diff, 0.5)
        else:
            loss_ph += w_pho[k] * charbonnier(diff, 1e-3)

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

        if LOSS_MODE == 0:
            loss_p += w_prot[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_p += w_prot[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_prot[k] + EPS)
            loss_p += w_prot[k] * pseudo_huber(diff, 0.5)
        else:
            loss_p += w_prot[k] * charbonnier(diff, 1e-3)

    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        R_t = Y[t_idx, start]
        R_b = Y[rna_base_idx, start]

        pred_fc = (R_t if R_t > EPS else EPS) / (R_b if R_b > EPS else EPS)

        diff = obs_rna[k] - pred_fc

        if LOSS_MODE == 0:
            loss_r += w_rna[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_r += w_rna[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_rna[k] + EPS)
            loss_r += w_rna[k] * pseudo_huber(diff, 0.5)
        else:
            loss_r += w_rna[k] * charbonnier(diff, 1e-3)

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

        if LOSS_MODE == 0:
            loss_ph += w_pho[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_ph += w_pho[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_pho[k] + EPS)
            loss_ph += w_pho[k] * pseudo_huber(diff, 0.5)
        else:
            loss_ph += w_pho[k] * charbonnier(diff, 1e-3)

    return loss_p, loss_r, loss_ph

LOSS_FN = loss_function_comb if MODEL == 2 else loss_function_noncomb