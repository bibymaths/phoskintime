"""Compute scalar loss values for protein, RNA, and phospho observations from simulated trajectories; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config."""

import numpy as np
from numba import njit
from networkmodel.config import MODEL, LOSS_MODE

EPS = 1e-9


@njit(fastmath=True, cache=True, nogil=True)
def sq(diff):
    """Compute squared loss
    
    Args:
        diff: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    return diff * diff


@njit(fastmath=True, cache=True, nogil=True)
def huber(diff, delta=1.0):
    """Compute Huber loss
    
    Args:
        diff: Input value used by this routine.
        delta: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    a = diff if diff >= 0.0 else -diff
    if a <= delta:
        return 0.5 * diff * diff
    return delta * (a - 0.5 * delta)


@njit(fastmath=True, cache=True, nogil=True)
def pseudo_huber(diff, delta=1.0):
    """Compute pseudo-Huber loss
    
    Args:
        diff: Input value used by this routine.
        delta: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    x = diff / delta
    return (delta * delta) * ((1.0 + x * x) ** 0.5 - 1.0)


@njit(fastmath=True, cache=True, nogil=True)
def charbonnier(diff, eps=1e-3):
    """Compute Charbonnier loss
    
    Args:
        diff: Input value used by this routine.
        eps: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    return (diff * diff + eps * eps) ** 0.5 - eps


@njit(fastmath=True, cache=True, nogil=True)
def log_cosh(diff):
    """Compute log-cosh loss
    
    Args:
        diff: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    s = np.abs(diff)
    if s > 20.0:
        # Avoid overflow: log(cosh(s)) ~= s - log(2) for large s
        return s - 0.69314718056
    return np.log(np.cosh(diff))


@njit(fastmath=True, cache=True, nogil=True)
def cauchy_loss(diff, c=1.0):
    """Compute Cauchy loss
    
    Args:
        diff: Input value used by this routine.
        c: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    return np.log(1.0 + (diff / c) ** 2)


@njit(fastmath=True, cache=True, nogil=True)
def poisson_scaled_mse(diff, pred_val, eps=1e-6):
    """Compute Poisson-scaled mean squared error
    
    Args:
        diff: Input value used by this routine.
        pred_val: Input value used by this routine.
        eps: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    # Weight = 1 / (Intensity + eps)
    # Loss = (Obs - Pred)^2 / Pred
    return (diff * diff) / (np.abs(pred_val) + eps)


@njit(fastmath=True, cache=True, nogil=True)
def geman_mcclure(diff, delta=1.0):
    """Compute Geman-McClure loss
    
    Args:
        diff: Input value used by this routine.
        delta: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    x2 = diff * diff
    return x2 / (x2 + delta * delta)


@njit(fastmath=True, cache=True, nogil=True)
def loss_function_noncomb(
        Y,
        p_prot, t_prot, obs_prot, w_prot,
        p_rna, t_rna, obs_rna, w_rna,
        p_pho, s_pho, t_pho, obs_pho, w_pho,
        prot_map,
        prot_base_idx, rna_base_idx, pho_base_idx
):
    """Compute multimodal loss for non-combinatorial state layouts
    
    Args:
        Y: Input value used by this routine.
        p_prot: Input value used by this routine.
        t_prot: Input value used by this routine.
        obs_prot: Input value used by this routine.
        w_prot: Input value used by this routine.
        p_rna: Input value used by this routine.
        t_rna: Input value used by this routine.
        obs_rna: Input value used by this routine.
        w_rna: Input value used by this routine.
        p_pho: Input value used by this routine.
        s_pho: Input value used by this routine.
        t_pho: Input value used by this routine.
        obs_pho: Input value used by this routine.
        w_pho: Input value used by this routine.
        prot_map: Input value used by this routine.
        prot_base_idx: Input value used by this routine.
        rna_base_idx: Input value used by this routine.
        pho_base_idx: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        n_sites = prot_map[p_idx, 1]

        # Calculate Total Protein = Unphosphorylated + Sum(Phosphorylated)
        tot_t = Y[t_idx, start + 1]
        tot_b = Y[prot_base_idx, start + 1]
        for s in range(n_sites):
            tot_t += Y[t_idx, start + 2 + s]
            tot_b += Y[prot_base_idx, start + 2 + s]

        # Calculate Fold Change (Pred)
        pred_fc = (tot_t if tot_t > EPS else EPS) / (tot_b if tot_b > EPS else EPS)

        diff = obs_prot[k] - pred_fc

        # Apply selected error metric
        if LOSS_MODE == 0:
            loss_p += w_prot[k] * sq(diff)
        elif LOSS_MODE == 1:
            loss_p += w_prot[k] * huber(diff, 0.5)
        elif LOSS_MODE == 2:
            diff = np.log(diff + EPS) - np.log(obs_prot[k] + EPS)
            loss_p += w_prot[k] * pseudo_huber(diff, 0.5)
        elif LOSS_MODE == 3:
            loss_p += w_prot[k] * log_cosh(diff)
        elif LOSS_MODE == 4:
            loss_p += w_prot[k] * cauchy_loss(diff, 1.0)
        elif LOSS_MODE == 5:
            loss_p += w_prot[k] * poisson_scaled_mse(diff, pred_fc, 1e-6)
        elif LOSS_MODE == 6:
            loss_p += w_prot[k] * geman_mcclure(diff, 1.0)
        else:
            loss_p += w_prot[k] * charbonnier(diff, 1e-3)

    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        # RNA is the first state in the block
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
        elif LOSS_MODE == 3:
            loss_r += w_rna[k] * log_cosh(diff)
        elif LOSS_MODE == 4:
            loss_r += w_rna[k] * cauchy_loss(diff, 1.0)
        elif LOSS_MODE == 5:
            loss_r += w_rna[k] * poisson_scaled_mse(diff, pred_fc, 1e-6)
        elif LOSS_MODE == 6:
            loss_r += w_rna[k] * geman_mcclure(diff, 1.0)
        else:
            loss_r += w_rna[k] * charbonnier(diff, 1e-3)

    loss_ph = 0.0
    for k in range(p_pho.size):
        p_idx = p_pho[k]
        s_idx = s_pho[k]
        t_idx = t_pho[k]
        start = prot_map[p_idx, 0]

        # Phospho site j is at offset: start + 2 + j
        ph_t = Y[t_idx, start + 2 + s_idx]
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
        elif LOSS_MODE == 3:
            loss_ph += w_pho[k] * log_cosh(diff)
        elif LOSS_MODE == 4:
            loss_ph += w_pho[k] * cauchy_loss(diff, 1.0)
        elif LOSS_MODE == 5:
            loss_ph += w_pho[k] * poisson_scaled_mse(diff, pred_fc, 1e-6)
        elif LOSS_MODE == 6:
            loss_ph += w_pho[k] * geman_mcclure(diff, 1.0)
        else:
            loss_ph += w_pho[k] * charbonnier(diff, 1e-3)

    return loss_p, loss_r, loss_ph


@njit(fastmath=True, cache=True, nogil=True)
def loss_function_comb(
        Y,
        p_prot, t_prot, obs_prot, w_prot,
        p_rna, t_rna, obs_rna, w_rna,
        p_pho, s_pho, t_pho, obs_pho, w_pho,
        prot_map,
        prot_base_idx, rna_base_idx, pho_base_idx
):
    """Compute multimodal loss for combinatorial state layouts
    
    Args:
        Y: Input value used by this routine.
        p_prot: Input value used by this routine.
        t_prot: Input value used by this routine.
        obs_prot: Input value used by this routine.
        w_prot: Input value used by this routine.
        p_rna: Input value used by this routine.
        t_rna: Input value used by this routine.
        obs_rna: Input value used by this routine.
        w_rna: Input value used by this routine.
        p_pho: Input value used by this routine.
        s_pho: Input value used by this routine.
        t_pho: Input value used by this routine.
        obs_pho: Input value used by this routine.
        w_pho: Input value used by this routine.
        prot_map: Input value used by this routine.
        prot_base_idx: Input value used by this routine.
        rna_base_idx: Input value used by this routine.
        pho_base_idx: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    loss_p = 0.0
    for k in range(p_prot.size):
        p_idx = p_prot[k]
        t_idx = t_prot[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]  # nstates = 2^n_sites
        p0 = start + 1

        tot_t = 0.0
        tot_b = 0.0
        # Sum all combinatorial states to get total protein
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
        elif LOSS_MODE == 3:
            loss_p += w_prot[k] * log_cosh(diff)
        elif LOSS_MODE == 4:
            loss_p += w_prot[k] * cauchy_loss(diff, 1.0)
        elif LOSS_MODE == 5:
            loss_p += w_prot[k] * poisson_scaled_mse(diff, pred_fc, 1e-6)
        elif LOSS_MODE == 6:
            loss_p += w_prot[k] * geman_mcclure(diff, 1.0)
        else:
            loss_p += w_prot[k] * charbonnier(diff, 1e-3)

    loss_r = 0.0
    for k in range(p_rna.size):
        p_idx = p_rna[k]
        t_idx = t_rna[k]
        start = prot_map[p_idx, 0]

        # RNA is still just the first element
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
        elif LOSS_MODE == 3:
            loss_r += w_rna[k] * log_cosh(diff)
        elif LOSS_MODE == 4:
            loss_r += w_rna[k] * cauchy_loss(diff, 1.0)
        elif LOSS_MODE == 5:
            loss_r += w_rna[k] * poisson_scaled_mse(diff, pred_fc, 1e-6)
        elif LOSS_MODE == 6:
            loss_r += w_rna[k] * geman_mcclure(diff, 1.0)
        else:
            loss_r += w_rna[k] * charbonnier(diff, 1e-3)

    loss_ph = 0.0
    for k in range(p_pho.size):
        p_idx = p_pho[k]
        j = s_pho[k]  # site index (0..n-1)
        t_idx = t_pho[k]
        start = prot_map[p_idx, 0]
        nstates = prot_map[p_idx, 1]
        p0 = start + 1

        ph_t = 0.0
        ph_b = 0.0
        # Iterate all states, check if site j is phosphorylated using bitwise AND
        # State m corresponds to binary pattern of phosphorylation.
        # e.g., if m=5 (binary 101), site 0 and site 2 are phosphorylated.
        for m in range(nstates):
            if (m >> j) & 1:
                ph_t += Y[t_idx, p0 + m]
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
        elif LOSS_MODE == 3:
            loss_ph += w_pho[k] * log_cosh(diff)
        elif LOSS_MODE == 4:
            loss_ph += w_pho[k] * cauchy_loss(diff, 1.0)
        elif LOSS_MODE == 5:
            loss_ph += w_pho[k] * poisson_scaled_mse(diff, pred_fc, 1e-6)
        elif LOSS_MODE == 6:
            loss_ph += w_pho[k] * geman_mcclure(diff, 1.0)
        else:
            loss_ph += w_pho[k] * charbonnier(diff, 1e-3)

    return loss_p, loss_r, loss_ph


# Dispatch the correct loss function based on configuration
LOSS_FN = loss_function_comb if MODEL == 2 else loss_function_noncomb
