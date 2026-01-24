"""
High-Performance Loss Functions Module.

This module implements the objective functions used to quantify the discrepancy
between model predictions ($Y$) and experimental data (Protein, RNA, Phospho).

**Key Features:**
1.  **JIT Compilation:** All functions are decorated with `@njit` from Numba to
    enable C-like performance, which is critical when the loss function is called
    thousands of times during optimization.
2.  **Multi-Modal Support:** Handles RNA, Total Protein, and Phospho-site data
    simultaneously.
3.  **Robust Error Metrics:** Supports multiple loss modes (Squared Error, Huber,
    Pseudo-Huber, Charbonnier) to handle outliers in biological data.
4.  **Topology Awareness:** Includes separate logic for standard models (linear states)
    and combinatorial models (bitwise state aggregation).


"""

import numpy as np
from numba import njit
from global_model.config import MODEL, LOSS_MODE

EPS = 1e-9


@njit(fastmath=True, cache=True, nogil=True)
def sq(diff):
    """Standard Squared Error: (y - y_pred)^2."""
    return diff * diff


@njit(fastmath=True, cache=True, nogil=True)
def huber(diff, delta=1.0):
    """
    Huber Loss.
    Behaves like Squared Error for small errors (<= delta) and Absolute Error
    for large errors. Robust to outliers.
    """
    a = diff if diff >= 0.0 else -diff
    if a <= delta:
        return 0.5 * diff * diff
    return delta * (a - 0.5 * delta)


@njit(fastmath=True, cache=True, nogil=True)
def pseudo_huber(diff, delta=1.0):
    """
    Pseudo-Huber Loss.
    A smooth approximation of Huber loss that is differentiable everywhere.
    """
    x = diff / delta
    return (delta * delta) * ((1.0 + x * x) ** 0.5 - 1.0)


@njit(fastmath=True, cache=True, nogil=True)
def charbonnier(diff, eps=1e-3):
    """
    Charbonnier Loss (differentiable L1).
    $\sqrt{diff^2 + \epsilon^2}$. Very robust to outliers.
    """
    return (diff * diff + eps * eps) ** 0.5 - eps


@njit(fastmath=True, cache=True, nogil=True)
def log_cosh(diff):
    """
    Log-Hyperbolic Cosine Loss.
    Approx: x^2/2 for small x, abs(x) - log(2) for large x.
    Smoother than Huber; helps gradients flow better near zero.
    """
    s = np.abs(diff)
    if s > 20.0:
        # Avoid overflow: log(cosh(s)) ~= s - log(2) for large s
        return s - 0.69314718056
    return np.log(np.cosh(diff))


@njit(fastmath=True, cache=True, nogil=True)
def cauchy_loss(diff, c=1.0):
    """
    Cauchy (Lorentzian) Loss.
    Scales logarithmically for large errors.
    Extremely robust to outliers; the influence of an outlier tends to zero.
    """
    return np.log(1.0 + (diff / c) ** 2)


@njit(fastmath=True, cache=True, nogil=True)
def poisson_scaled_mse(diff, pred_val, eps=1e-6):
    """
    Poisson-Scaled MSE.
    Weights the error by the inverse of the predicted intensity.
    Penalizes relative error more heavily at low values and allows more variance at high values.
    """
    # Weight = 1 / (Intensity + eps)
    # Loss = (Obs - Pred)^2 / Pred
    return (diff * diff) / (np.abs(pred_val) + eps)


@njit(fastmath=True, cache=True, nogil=True)
def geman_mcclure(diff, delta=1.0):
    """
    Geman-McClure Loss.
    Soft-saturating loss. As error -> infinity, loss -> constant.
    This strictly limits the maximum penalty any single data point can exert.
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
    """
    Loss function for Models 0, 1, and 4 (Non-Combinatorial).

    In these models, the state vector is linear:
    [RNA, Unphos, Phos_site1, Phos_site2, ...]

    Calculates:
    1.  **Protein:** (Unphos + Sum(Phos_states)) / Baseline
    2.  **RNA:** RNA_state / Baseline
    3.  **Phospho:** Phos_state_j / Baseline

    Args:
        Y (np.ndarray): Simulation trajectory (n_times x n_states).
        *_prot, *_rna, *_pho: Arrays of indices (protein index, time index, etc.),
                              observations, and weights for each data type.
        prot_map (np.ndarray): Lookup table for state offsets.
        *_base_idx: Index of the normalization baseline timepoint (usually t=0).

    Returns:
        tuple: (loss_protein, loss_rna, loss_phospho)
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
    """
    Loss function for Model 2 (Combinatorial).



    In this model, states represent all $2^n$ combinations of phosphorylation.
    To calculate observables, we must aggregate these states:
    1.  **Protein:** Sum of all $2^n$ states.
    2.  **Phospho Site j:** Sum of all states where the $j$-th bit is 1.

    Args:
        Y (np.ndarray): Simulation trajectory.
        ... (same as noncomb) ...
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
