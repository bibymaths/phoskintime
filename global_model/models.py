"""
Kinetic Model Kernels (Right-Hand Side Definitions).

This module contains the JIT-compiled kernels that define the system of Ordinary Differential Equations (ODEs)
for different biological kinetic assumptions.

**Supported Models:**
1.  **Distributive (Model 0):** Sites on the same protein are phosphorylated independently.
    State space scales linearly with sites ($N_{sites} + 2$).
2.  **Sequential (Model 1):** Phosphorylation occurs in a strict order ($P_0 \to P_1 \to \dots$).
    Useful for processive kinases. State space is linear.
3.  **Combinatorial (Model 2):** Every possible phosphorylation pattern is a distinct state.
    State space scales exponentially ($2^{N_{sites}} + 1$). Handles complex logic like logic gates.
4.  **Saturating (Model 4):** Uses Michaelis-Menten kinetics for translation and phosphorylation.
    Prevents unbiological rate explosions at high concentrations.


"""

import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# Helper: Optimized Activation Logic
# -----------------------------------------------------------------------------
@njit(fastmath=True, cache=True, nogil=True, inline='always')
def calculate_synthesis_rate(Ai, tf_scale, u_raw):
    """
    Computes the transcription rate based on Transcription Factor (TF) input.

    Uses a rational (Hill-like) function that is numerically stable and bounded.



    Logic:
    - **Soft-Clipping:** The raw input `u` is first mapped to (-1, 1) to prevent
      numerical overflow/instability using $u_{norm} = u / (1 + |u|)$.
    - **Activation (u > 0):** Rate increases from $A_i$ to $A_i \times (1 + tf\_scale)$.
    - **Repression (u < 0):** Rate decreases from $A_i$ to $A_i / (1 + tf\_scale)$.

    Args:
        Ai (float): Basal synthesis rate.
        tf_scale (float): Maximum fold-change factor.
        u_raw (float): Raw total TF input (weighted sum of regulators).

    Returns:
        float: The calculated synthesis rate.
    """
    # 1. Squash input to (-1, 1) to prevent numerical instability
    # This acts as a soft-clipping mechanism.
    u = u_raw / (1.0 + np.abs(u_raw))

    if u >= 0.0:
        # Activation: Rate increases but hits a ceiling
        # Formula: Ai * (1 + (scale * u) / (1 + u))
        # As u -> 1, Rate -> Ai * (1 + scale/2)
        term = (tf_scale * u) / (1.0 + u + 1e-6)
        return Ai * (1.0 + term)
    else:
        # Repression: Rate decreases but hits a floor > 0
        # Formula: Ai / (1 + scale * |u|)
        # As u -> -1, Rate -> Ai / (1 + scale)
        denom = 1.0 + tf_scale * np.abs(u)
        return Ai / denom


# -----------------------------------------------------------------------------
# Model 4: Saturating (Michaelis-Menten)
# -----------------------------------------------------------------------------
@njit(fastmath=True, cache=True, nogil=True)
def saturating_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                   offset_y, offset_s, n_sites):
    """
    Right-Hand Side for Model 4: Saturating Kinetics.



    Unlike other models that use Mass Action kinetics (Rate = k * [S]), this model
    uses Michaelis-Menten forms (Rate = Vmax * [S] / (Km + [S])) for Translation
    and Phosphorylation. This prevents "runaway" kinetics where high protein
    concentrations lead to infinitely fast reactions.

    Physics:
    - **Translation:** Limited by ribosome availability.
    - **Phosphorylation:** Limited by enzyme (kinase) availability relative to substrate.
    """
    N = A_i.shape[0]
    K_SAT = 1.0  # Saturation constant (normalized units)

    for i in range(N):
        y_start = offset_y[i]
        idx_R = y_start
        idx_P = y_start + 1

        s_start = offset_s[i]
        ns = n_sites[i]
        base = y_start + 2

        R = y[idx_R]
        P = y[idx_P]

        # 1. Rational Transcription
        synth = calculate_synthesis_rate(A_i[i], tf_scale, TF_inputs[i])

        # 2. mRNA Dynamics
        dy[idx_R] = synth - B_i[i] * R

        # 3. Saturating Translation (Ribosome limit)
        # Rate = Vmax * R / (Km + R) -> Here Km=1.0 relative to normalized data
        trans_rate = (C_i[i] * R) / (1.0 + R)

        prot_deg = D_i[i] * P

        if ns == 0:
            dy[idx_P] = trans_rate - prot_deg
        else:
            sum_S_flux = 0.0
            sum_back = 0.0

            # 4. Saturating Phosphorylation (Kinase limit)
            # Prevents stiff derivatives when P is large
            for j in range(ns):
                si = s_start + j
                yi = base + j

                s_rate_const = S_all[si]
                ps_val = y[yi]

                # Forward Rate = k * P / (1 + P)
                forward_flux = (s_rate_const * P) / (1.0 + P)

                # Backward Rate (Dephosph) - Linear or Saturating
                # Linear is usually stable enough for phosphatase (high capacity)
                backward_flux = E_i[i] * ps_val

                sum_S_flux += forward_flux
                sum_back += backward_flux

                # Phospho-site state equation
                Dpi = Dp_i[si]
                # Note: Includes base protein degradation (Di) + specific phospho-decay (Dpi)
                dy[yi] = forward_flux - (Dpi + D_i[i]) * ps_val - backward_flux

            # Unphosph protein equation
            dy[idx_P] = trans_rate - prot_deg - sum_S_flux + sum_back


@njit(fastmath=True, cache=True, nogil=True)
def distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                     offset_y, offset_s, n_sites):
    """
    Right-Hand Side for Model 0: Distributive (Independent) Binding.

    Assumes that the phosphorylation status of one site does not affect others.
    We track the Unphosphorylated species $P$ and $N_{sites}$ distinct mono-phosphorylated species.
    This is a simplification that ignores multi-phosphorylated states but captures individual site dynamics efficiently.
    """
    N = A_i.shape[0]

    for i in range(N):
        y_start = offset_y[i]
        idx_R = y_start
        idx_P = y_start + 1

        s_start = offset_s[i]
        ns = n_sites[i]
        base = y_start + 2  # first phospho state index in y

        R = y[idx_R]
        P = y[idx_P]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]
        Ei = E_i[i]

        # Calculate synthesis rate using optimized helper
        u = TF_inputs[i]
        synth = calculate_synthesis_rate(Ai, tf_scale, u)

        # mRNA
        dy[idx_R] = synth - Bi * R

        if ns == 0:
            # protein only
            dy[idx_P] = Ci * R - Di * P
        else:
            sum_S = 0.0
            sum_back = 0.0

            # phospho states
            for j in range(ns):
                si = s_start + j
                yi = base + j

                s_rate = S_all[si]
                ps_val = y[yi]

                sum_S += s_rate
                sum_back += Ei * ps_val

                Dpi = Dp_i[si]

                # Added protein degradation term to each phospho state decay
                # Explanation - Phosphorylated protein is still the same protein, but has a different state
                # Decay = (Dephosphorylation Rate + Specific Decay + Global Decay)
                dy[yi] = s_rate * P - (Ei + Dpi + Di) * ps_val

            # unphosph protein
            dy[idx_P] = Ci * R - (Di + sum_S) * P + sum_back


@njit(fastmath=True, cache=True, nogil=True)
def sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                   offset_y, offset_s, n_sites):
    """
    Right-Hand Side for Model 1: Sequential Binding.

    Assumes phosphorylation must happen in a specific order:
    $P_0 \xrightarrow{k_0} P_1 \xrightarrow{k_1} P_2 \dots \xrightarrow{k_{n-1}} P_n$

    Dephosphorylation is assumed to be distributive (any state can revert to the previous one).
    This model is suitable for processive kinases or mechanisms with strict steric requirements.
    """
    N = A_i.shape[0]

    for i in range(N):
        y_start = offset_y[i]
        idx_R = y_start
        idx_P0 = y_start + 1

        s_start = offset_s[i]
        ns = n_sites[i]
        base = y_start + 2  # P1 at base+0

        R = y[idx_R]
        P0 = y[idx_P0]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]
        Ei = E_i[i]

        # Calculate synthesis rate using optimized helper
        u = TF_inputs[i]
        synth = calculate_synthesis_rate(Ai, tf_scale, u)

        # mRNA
        dy[idx_R] = synth - Bi * R

        if ns == 0:
            dy[idx_P0] = Ci * R - Di * P0
            continue

        # --- P0 (unphosph) ---
        # Consumed by k0 (first step), produced by dephosphorylation of P1
        k0 = S_all[s_start + 0]
        P1 = y[base + 0]
        dy[idx_P0] = Ci * R - Di * P0 - k0 * P0 + Ei * P1

        if ns == 1:
            # --- last state is P1 ---
            Dp1 = Dp_i[s_start + 0]

            # Added protein degradation term to each phospho state decay
            # Explanation - Phosphorylated protein is still the same protein, but has a different state
            dy[base + 0] = k0 * P0 - (Ei + Dp1 + Di) * P1
            continue

        # --- P1 (first phospho) handled separately to avoid branch in loop ---
        k1 = S_all[s_start + 1]
        P2 = y[base + 1]
        Dp1 = Dp_i[s_start + 0]

        # Added protein degradation term to each phospho state decay
        dy[base + 0] = k0 * P0 + Ei * P2 - (k1 + Ei + Dp1 + Di) * P1

        # --- middle states: P2..P(ns-1) ---
        # indices base+1 .. base+(ns-2)
        for j in range(1, ns - 1):
            idx = base + j  # P(j+1)
            Pj = y[idx]

            k_prev = S_all[s_start + j]  # forward from previous -> current
            k_next = S_all[s_start + j + 1]  # forward from current -> next

            P_prev = y[idx - 1]
            P_next = y[idx + 1]

            Dpj = Dp_i[s_start + j]  # j=1 corresponds to P2, etc.

            # Flux in from left, Flux in from right (dephos), Flux out to right, Flux out to left (dephos), Decay
            dy[idx] = k_prev * P_prev + Ei * P_next - (k_next + Ei + Dpj + Di) * Pj

        # --- last state: Pns (index base + ns - 1) ---
        idx_last = base + (ns - 1)
        Plast = y[idx_last]
        k_last = S_all[s_start + (ns - 1)]
        Pprev = y[idx_last - 1]
        Dp_last = Dp_i[s_start + (ns - 1)]

        # Added protein degradation term to each phospho state decay
        dy[idx_last] = k_last * Pprev - (Ei + Dp_last + Di) * Plast


@njit(cache=True, nogil=True)
def _bit_index_from_lsb(lsb):
    """
    Fast conversion of a Least Significant Bit (power of 2) to its integer index.
    e.g., 4 (binary 100) -> 2.
    """
    j = 0
    while lsb > 1:
        lsb >>= 1
        j += 1
    return j


@njit(fastmath=True, cache=True, nogil=True)
def combinatorial_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs, S_cache, jb,
        offset_y, offset_s,
        n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n
):
    """
    Right-Hand Side for Model 2: Combinatorial Binding.



    Tracks all $2^N$ possible states. This allows for complex logic (e.g., "Site A facilitates Site B").

    The state transition logic is split into:
    1.  **Dephosphorylation (Implicit Graph):** Iterates over all states `m`. For every set bit in `m`, 
        calculates decay to `m ^ lsb`.
    2.  **Phosphorylation (Explicit Graph):** Uses pre-calculated `trans_*` arrays to apply forward 
        transitions based on available sites.
    """
    N = A_i.shape[0]
    jb_loc = jb  # local binding helps Numba

    for i in range(N):
        y_start = offset_y[i]
        s_start = offset_s[i]
        ns = n_sites[i]

        idx_R = y_start
        idx_P0 = y_start + 1

        R = y[idx_R]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]
        Ei = E_i[i]

        # Calculate synthesis rate using optimized helper
        u = TF_inputs[i]
        synth = calculate_synthesis_rate(Ai, tf_scale, u)

        # mRNA
        dy[idx_R] = synth - Bi * R

        # No sites: simple protein production/decay
        if ns == 0:
            P0 = y[idx_P0]
            dy[idx_P0] = Ci * R - Di * P0
            continue

        nstates = n_states[i]

        # translation adds to the totally unphosphorylated state (mask=0)
        dy[idx_P0] += Ci * R

        # --- Decay & Dephosphorylation Loop ---
        base = idx_P0

        # m = 0 (Unphos) state only has basic decay
        P0 = y[base]
        dy[base] += -Di * P0

        # m > 0 states: dephosph transitions + per-site decay
        for m in range(1, nstates):
            Pm = y[base + m]
            if Pm == 0.0:
                continue

            mm = m
            dp_rate = 0.0

            # Iterate over set bits in state m to find decay paths
            while mm != 0:
                lsb = mm & -mm  # Extract lowest set bit
                mm -= lsb  # Remove it for next iter

                j = _bit_index_from_lsb(lsb)  # 0..ns-1
                to = m ^ lsb  # Target state (current state minus one phospho group)

                # dephosph transition: m -> to at rate Ei * Pm (per set bit)
                flux = Ei * Pm
                dy[base + m] -= flux
                dy[base + to] += flux

                # per-site decay contribution (sink)
                # Added protein degradation term to each phospho state decay
                # Explanation - Phosphorylated protein is still the same protein, but has a different state
                dp_rate += Dp_i[s_start + j] + Di

            # apply summed per-site decay to this mask
            dy[base + m] -= dp_rate * Pm

        # --- Phosphorylation Loop (Forward Transitions) ---
        # Uses pre-calculated sparse graph structure
        off = trans_off[i]
        ntr = trans_n[i]
        for k in range(ntr):
            frm = trans_from[off + k]
            to = trans_to[off + k]
            j = trans_site[off + k]

            # Rate depends on time bucket 'jb_loc'
            rate = S_cache[s_start + j, jb_loc]
            flux = rate * y[base + frm]

            dy[base + frm] -= flux
            dy[base + to] += flux


def build_random_transitions(idx):
    """
    Precompute random phosphorylation transitions for all proteins (Model 2 Setup).

    Constructs the sparse adjacency list for the hypercube state graph.

    Returns arrays for Numba:
      trans_from, trans_to, trans_site  (flattened)
      trans_off[i], trans_n[i] per protein i

    Interpretation:
      for protein i:
        transitions are in slice [trans_off[i] : trans_off[i]+trans_n[i]]
        each transition uses site index 'j' (0..ns-1) to pick rate S_all[s_start + j]
    """
    trans_from = []
    trans_to = []
    trans_site = []
    trans_off = np.zeros(idx.N, dtype=np.int32)
    trans_n = np.zeros(idx.N, dtype=np.int32)

    cur = 0
    for i in range(idx.N):
        ns = int(idx.n_sites[i])
        trans_off[i] = cur

        if ns == 0:
            trans_n[i] = 0
            continue

        nstates = 1 << ns
        for m in range(nstates):
            for j in range(ns):
                # If bit j is NOT set in m, we can transition to m | (1<<j)
                if (m & (1 << j)) == 0:
                    mp = m | (1 << j)
                    trans_from.append(m)
                    trans_to.append(mp)
                    trans_site.append(j)

        n_i = len(trans_from) - cur
        trans_n[i] = n_i
        cur += n_i

    return (
        np.asarray(trans_from, dtype=np.int32),
        np.asarray(trans_to, dtype=np.int32),
        np.asarray(trans_site, dtype=np.int32),
        trans_off,
        trans_n,
    )