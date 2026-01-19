import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True, nogil=True)
def distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                     offset_y, offset_s, n_sites):
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

        # Use Linear coupling
        # This maps u=(-1 to 1) to a linear Fold Change range.
        # Example: tf_scale = 0.8
        # Max Repression: 1.0 + 0.8*(-1) = 0.20x
        # Max Activation: 1.0 + 0.8*(+1) = 1.80x
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))
        # synth = Ai * (1.0 + tf_scale * u)

        # Use Exponential coupling
        # This maps u=(-1 to 1) to a Fold Change range.
        # Example: tf_scale = 2.0
        # Max Repression: exp(-2) = 0.13x
        # Max Activation: exp(+2) = 7.38x
        u = TF_inputs[i]
        u = u / (1.0 + np.abs(u))
        synth = Ai * np.exp(tf_scale * u)

        # Use Logistic (Sigmoid) coupling
        # This creates a bounded "Switch-like" behavior.
        # At u=0 (no input), rate = Ai (Basal).
        # As u -> +1, rate saturates at 2*Ai (Max activation).
        # As u -> -1, rate approaches 0 (Complete repression).
        # Higher tf_scale makes the switch sharper.
        # u = TF_inputs[i]
        # val = tf_scale * TF_inputs[i]
        # synth = Ai * (2.0 / (1.0 + np.exp(-val)))

        # Use Rational (Hill-like) coupling
        # This models saturation: "Diminishing Returns"
        # Activation (u > 0): Rate grows but hits a ceiling at (1 + tf_scale) * Ai
        # Repression (u < 0): Rate drops but hits a floor.
        # This prevents the "infinite energy" problem of Exponential coupling.
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))  # Keep the squash
        # if u >= 0:
        #     # Activation: 1 + (scale * u) / (1 + u)
        #     synth = Ai * (1.0 + (tf_scale * u) / (1.0 + u + 1e-6))
        # else:
        #     # Repression: 1 / (1 + scale * |u|)
        #     synth = Ai / (1.0 + tf_scale * np.abs(u))

        # Use Power-Law coupling
        # This mimics high-sensitivity cooperative binding (Hill Coefficient).
        # If tf_scale (n) > 1, the gene ignores low signals but spikes at high signals.
        # u is the base fold-change driver.
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))
        #
        # # Map (-1, 1) to a base (0.1, 10)
        # base = (1.0 + u) / (1.0 - u + 1e-6)  # Maps -1->0, 0->1, 1->Inf
        #
        # # Apply power law sensitivity
        # # tf_scale acts as the sensitivity exponent
        # synth = Ai * np.power(base, tf_scale)

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
                dy[yi] = s_rate * P - (Ei + Dpi + Di) * ps_val

            # unphosph protein
            dy[idx_P] = Ci * R - (Di + sum_S) * P + sum_back


@njit(fastmath=True, cache=True, nogil=True)
def sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                   offset_y, offset_s, n_sites):
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

        # Use Linear coupling
        # This maps u=(-1 to 1) to a linear Fold Change range.
        # Example: tf_scale = 0.8
        # Max Repression: 1.0 + 0.8*(-1) = 0.20x
        # Max Activation: 1.0 + 0.8*(+1) = 1.80x
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))
        # synth = Ai * (1.0 + tf_scale * u)

        # Use Exponential coupling
        # This maps u=(-1 to 1) to a Fold Change range.
        # Example: tf_scale = 2.0
        # Max Repression: exp(-2) = 0.13x
        # Max Activation: exp(+2) = 7.38x
        u = TF_inputs[i]
        u = u / (1.0 + np.abs(u))
        synth = Ai * np.exp(tf_scale * u)

        # Use Logistic (Sigmoid) coupling
        # This creates a bounded "Switch-like" behavior.
        # At u=0 (no input), rate = Ai (Basal).
        # As u -> +1, rate saturates at 2*Ai (Max activation).
        # As u -> -1, rate approaches 0 (Complete repression).
        # Higher tf_scale makes the switch sharper.
        # u = TF_inputs[i]
        # val = tf_scale * TF_inputs[i]
        # synth = Ai * (2.0 / (1.0 + np.exp(-val)))

        # Use Rational (Hill-like) coupling
        # This models saturation: "Diminishing Returns"
        # Activation (u > 0): Rate grows but hits a ceiling at (1 + tf_scale) * Ai
        # Repression (u < 0): Rate drops but hits a floor.
        # This prevents the "infinite energy" problem of Exponential coupling.
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))  # Keep the squash
        # if u >= 0:
        #     # Activation: 1 + (scale * u) / (1 + u)
        #     synth = Ai * (1.0 + (tf_scale * u) / (1.0 + u + 1e-6))
        # else:
        #     # Repression: 1 / (1 + scale * |u|)
        #     synth = Ai / (1.0 + tf_scale * np.abs(u))

        # Use Power-Law coupling
        # This mimics high-sensitivity cooperative binding (Hill Coefficient).
        # If tf_scale (n) > 1, the gene ignores low signals but spikes at high signals.
        # u is the base fold-change driver.
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))
        #
        # # Map (-1, 1) to a base (0.1, 10)
        # base = (1.0 + u) / (1.0 - u + 1e-6)  # Maps -1->0, 0->1, 1->Inf
        #
        # # Apply power law sensitivity
        # # tf_scale acts as the sensitivity exponent
        # synth = Ai * np.power(base, tf_scale)


        # mRNA
        dy[idx_R] = synth - Bi * R

        if ns == 0:
            dy[idx_P0] = Ci * R - Di * P0
            continue

        # --- P0 (unphosph) ---
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
        # Explanation - Phosphorylated protein is still the same protein, but has a different state
        dy[base + 0] = k0 * P0 + Ei * P2 - (k1 + Ei + Dp1 + Di) * P1

        # --- middle states: P2..P(ns-1) ---
        # indices base+1 .. base+(ns-2)
        for j in range(1, ns - 1):
            idx = base + j          # P(j+1)
            Pj = y[idx]

            k_prev = S_all[s_start + j]       # forward from previous -> current
            k_next = S_all[s_start + j + 1]   # forward from current -> next

            P_prev = y[idx - 1]
            P_next = y[idx + 1]

            Dpj = Dp_i[s_start + j]  # j=1 corresponds to P2, etc.

            # Added protein degradation term to each phospho state decay
            # Explanation - Phosphorylated protein is still the same protein, but has a different state
            dy[idx] = k_prev * P_prev + Ei * P_next - (k_next + Ei + Dpj + Di) * Pj

        # --- last state: Pns (index base + ns - 1) ---
        idx_last = base + (ns - 1)
        Plast = y[idx_last]
        k_last = S_all[s_start + (ns - 1)]
        Pprev = y[idx_last - 1]
        Dp_last = Dp_i[s_start + (ns - 1)]

        # Added protein degradation term to each phospho state decay
        # Explanation - Phosphorylated protein is still the same protein, but has a different state
        dy[idx_last] = k_last * Pprev - (Ei + Dp_last + Di) * Plast

@njit(cache=True, nogil=True)
def _bit_index_from_lsb(lsb):
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

        # Use Linear coupling
        # This maps u=(-1 to 1) to a linear Fold Change range.
        # Example: tf_scale = 0.8
        # Max Repression: 1.0 + 0.8*(-1) = 0.20x
        # Max Activation: 1.0 + 0.8*(+1) = 1.80x
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))
        # synth = Ai * (1.0 + tf_scale * u)

        # Use Exponential coupling
        # This maps u=(-1 to 1) to a Fold Change range.
        # Example: tf_scale = 2.0
        # Max Repression: exp(-2) = 0.13x
        # Max Activation: exp(+2) = 7.38x
        u = TF_inputs[i]
        u = u / (1.0 + np.abs(u))
        synth = Ai * np.exp(tf_scale * u)

        # Use Logistic (Sigmoid) coupling
        # This creates a bounded "Switch-like" behavior.
        # At u=0 (no input), rate = Ai (Basal).
        # As u -> +1, rate saturates at 2*Ai (Max activation).
        # As u -> -1, rate approaches 0 (Complete repression).
        # Higher tf_scale makes the switch sharper.
        # u = TF_inputs[i]
        # val = tf_scale * TF_inputs[i]
        # synth = Ai * (2.0 / (1.0 + np.exp(-val)))

        # Use Rational (Hill-like) coupling
        # This models saturation: "Diminishing Returns"
        # Activation (u > 0): Rate grows but hits a ceiling at (1 + tf_scale) * Ai
        # Repression (u < 0): Rate drops but hits a floor.
        # This prevents the "infinite energy" problem of Exponential coupling.
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))  # Keep the squash
        # if u >= 0:
        #     # Activation: 1 + (scale * u) / (1 + u)
        #     synth = Ai * (1.0 + (tf_scale * u) / (1.0 + u + 1e-6))
        # else:
        #     # Repression: 1 / (1 + scale * |u|)
        #     synth = Ai / (1.0 + tf_scale * np.abs(u))

        # Use Power-Law coupling
        # This mimics high-sensitivity cooperative binding (Hill Coefficient).
        # If tf_scale (n) > 1, the gene ignores low signals but spikes at high signals.
        # u is the base fold-change driver.
        # u = TF_inputs[i]
        # u = u / (1.0 + np.abs(u))
        #
        # # Map (-1, 1) to a base (0.1, 10)
        # base = (1.0 + u) / (1.0 - u + 1e-6)  # Maps -1->0, 0->1, 1->Inf
        #
        # # Apply power law sensitivity
        # # tf_scale acts as the sensitivity exponent
        # synth = Ai * np.power(base, tf_scale)

        # RNA
        dy[idx_R] = synth - Bi * R

        # No sites: simple protein production/decay
        if ns == 0:
            P0 = y[idx_P0]
            dy[idx_P0] = Ci * R - Di * P0
            continue

        nstates = n_states[i]

        # translation into mask=0
        dy[idx_P0] += Ci * R

        # decay for all states
        base = idx_P0

        # m = 0 state uses Di
        P0 = y[base]
        dy[base] += -Di * P0

        # m > 0 states: dephosph transitions + per-site decay
        for m in range(1, nstates):
            Pm = y[base + m]
            if Pm == 0.0:
                continue

            mm = m
            dp_rate = 0.0

            while mm != 0:
                lsb = mm & -mm
                mm -= lsb

                j = _bit_index_from_lsb(lsb)  # 0..ns-1
                to = m ^ lsb

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

        # forward phosphorylation transitions
        off = trans_off[i]
        ntr = trans_n[i]
        for k in range(ntr):
            frm = trans_from[off + k]
            to  = trans_to[off + k]
            j   = trans_site[off + k]

            rate = S_cache[s_start + j, jb_loc]
            flux = rate * y[base + frm]

            dy[base + frm] -= flux
            dy[base + to]  += flux

def build_random_transitions(idx):
    """
    Precompute random phosphorylation transitions for all proteins.

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