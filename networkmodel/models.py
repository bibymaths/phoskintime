"""Define NumPy right-hand-side kernels for supported phosphorylation topologies; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules."""

import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# Helper: Optimized Activation Logic
# -----------------------------------------------------------------------------
@njit(fastmath=True, cache=True, nogil=True, inline='always')
def calculate_synthesis_rate(Ai, tf_scale, u_raw):
    """Calculate saturating transcriptional synthesis rate
    
    Args:
        Ai: Input value used by this routine.
        tf_scale: Input value used by this routine.
        u_raw: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
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
    """Evaluate the saturating topology right-hand side
    
    Args:
        y: Input value used by this routine.
        dy: Input value used by this routine.
        A_i: Input value used by this routine.
        B_i: Input value used by this routine.
        C_i: Input value used by this routine.
        D_i: Input value used by this routine.
        Dp_i: Input value used by this routine.
        E_i: Input value used by this routine.
        tf_scale: Input value used by this routine.
        TF_inputs: Input value used by this routine.
        S_all: Input value used by this routine.
        offset_y: Input value used by this routine.
        offset_s: Input value used by this routine.
        n_sites: Input value used by this routine.
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
    """Evaluate the distributive topology right-hand side
    
    Args:
        y: Input value used by this routine.
        dy: Input value used by this routine.
        A_i: Input value used by this routine.
        B_i: Input value used by this routine.
        C_i: Input value used by this routine.
        D_i: Input value used by this routine.
        Dp_i: Input value used by this routine.
        E_i: Input value used by this routine.
        tf_scale: Input value used by this routine.
        TF_inputs: Input value used by this routine.
        S_all: Input value used by this routine.
        offset_y: Input value used by this routine.
        offset_s: Input value used by this routine.
        n_sites: Input value used by this routine.
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
    """Evaluate the sequential topology right-hand side
    
    Args:
        y: Input value used by this routine.
        dy: Input value used by this routine.
        A_i: Input value used by this routine.
        B_i: Input value used by this routine.
        C_i: Input value used by this routine.
        D_i: Input value used by this routine.
        Dp_i: Input value used by this routine.
        E_i: Input value used by this routine.
        tf_scale: Input value used by this routine.
        TF_inputs: Input value used by this routine.
        S_all: Input value used by this routine.
        offset_y: Input value used by this routine.
        offset_s: Input value used by this routine.
        n_sites: Input value used by this routine.
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
    """Handle internal bit index from lsb"""
    j = 0
    while lsb > 1:
        lsb >>= 1
        j += 1
    return j


@njit(fastmath=True, cache=True, nogil=True)
def combinatorial_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs, S_rates,
        offset_y, offset_s,
        n_sites, n_states
):
    """Evaluate the combinatorial topology right-hand side
    
    Args:
        y: Input value used by this routine.
        dy: Input value used by this routine.
        A_i: Input value used by this routine.
        B_i: Input value used by this routine.
        C_i: Input value used by this routine.
        D_i: Input value used by this routine.
        Dp_i: Input value used by this routine.
        E_i: Input value used by this routine.
        tf_scale: Input value used by this routine.
        TF_inputs: Input value used by this routine.
        S_rates: Current per-site kinase signal vector.
        offset_y: Input value used by this routine.
        offset_s: Input value used by this routine.
        n_sites: Input value used by this routine.
        n_states: Input value used by this routine.
    """
    N = A_i.shape[0]
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
        # Enumerate the same hypercube edges in the same order as the former
        # dense transition arrays, but do not materialize O(2^n_sites*n_sites)
        # arrays.
        for m in range(nstates):
            for j in range(ns):
                bit = 1 << j
                if (m & bit) == 0:
                    to = m | bit
                    rate = S_rates[s_start + j]
                    flux = rate * y[base + m]

                    dy[base + m] -= flux
                    dy[base + to] += flux


def iter_random_transitions_for_sites(n_sites):
    """Yield combinatorial forward transitions for one protein lazily.

    The order is identical to the historical dense implementation: state mask
    first, then site index, yielding only unset-bit phosphorylation edges.
    """
    ns = int(n_sites)
    if ns <= 0:
        return
    nstates = 1 << ns
    for m in range(nstates):
        for j in range(ns):
            if (m & (1 << j)) == 0:
                yield m, m | (1 << j), j


def count_random_transitions_for_sites(n_sites):
    """Return the number of combinatorial forward transitions for n sites."""
    ns = int(n_sites)
    return 0 if ns <= 0 else ns * (1 << (ns - 1))


def build_random_transitions(idx, *, dense_threshold_sites=4):
    """Build small dense transition arrays for compatibility.

    Large proteins are represented by metadata only; callers that need all
    transitions should use :func:`iter_random_transitions_for_sites` instead.
    """
    trans_from = []
    trans_to = []
    trans_site = []
    trans_off = np.zeros(idx.N, dtype=np.int32)
    trans_n = np.zeros(idx.N, dtype=np.int32)
    dense_available = np.zeros(idx.N, dtype=np.bool_)

    cur = 0
    for i in range(idx.N):
        ns = int(idx.n_sites[i])
        trans_off[i] = cur
        trans_n[i] = count_random_transitions_for_sites(ns)
        if ns <= int(dense_threshold_sites):
            dense_available[i] = True
            for frm, to, site in iter_random_transitions_for_sites(ns):
                trans_from.append(frm)
                trans_to.append(to)
                trans_site.append(site)
            cur = len(trans_from)
        else:
            dense_available[i] = False

    return (
        np.asarray(trans_from, dtype=np.int32),
        np.asarray(trans_to, dtype=np.int32),
        np.asarray(trans_site, dtype=np.int32),
        trans_off,
        trans_n,
        dense_available,
    )
