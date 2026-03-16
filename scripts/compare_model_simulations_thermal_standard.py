import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit


# =============================================================================
# 1. Helper Functions
# =============================================================================
@njit(fastmath=True, cache=True, nogil=True)
def calculate_synthesis_rate(Ai, tf_scale, u_raw):
    u = u_raw / (1.0 + np.abs(u_raw))
    if u >= 0.0:
        return Ai * (1.0 + (tf_scale * u) / (1.0 + u + 1e-6))
    else:
        return Ai / (1.0 + tf_scale * np.abs(u))


@njit(fastmath=True, cache=True, nogil=True)
def calculate_folded_fraction(T, Tm, c_fold=0.8):
    return 1.0 / (1.0 + np.exp(c_fold * (T - Tm)))


@njit(cache=True, nogil=True)
def _bit_index_from_lsb(lsb):
    j = 0
    while lsb > 1:
        lsb >>= 1
        j += 1
    return j


# =============================================================================
# 2. Numba Kernels (Thermal-Integrated)
# =============================================================================
@njit(fastmath=True, nogil=True)
def distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                     offset_y, offset_s, n_sites, T, Tm_i, c_fold=0.8, k_unfold=4.0):
    for i in range(A_i.shape[0]):
        idx_R, idx_P = offset_y[i], offset_y[i] + 1
        s_start, ns, base = offset_s[i], n_sites[i], offset_y[i] + 2
        R, P = y[idx_R], y[idx_P]

        dy[idx_R] = calculate_synthesis_rate(A_i[i], tf_scale, TF_inputs[i]) - B_i[i] * R
        f_folded = calculate_folded_fraction(T, Tm_i[i], c_fold)
        Di_therm = D_i[i] * (1.0 + k_unfold * (1.0 - f_folded))
        P_active = P * f_folded

        sum_S = sum_back = 0.0
        for j in range(ns):
            yi, si = base + j, s_start + j
            s_rate, ps_val = S_all[si], y[yi]
            sum_S += s_rate
            sum_back += E_i[i] * ps_val
            Dpi_therm = Dp_i[si] * (1.0 + k_unfold * (1.0 - f_folded))
            dy[yi] = s_rate * P_active - (E_i[i] + Dpi_therm + Di_therm) * ps_val

        dy[idx_P] = C_i[i] * R - Di_therm * P - sum_S * P_active + sum_back


@njit(fastmath=True, nogil=True)
def sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                   offset_y, offset_s, n_sites, T, Tm_i, c_fold=0.8, k_unfold=4.0):
    for i in range(A_i.shape[0]):
        idx_R, idx_P0 = offset_y[i], offset_y[i] + 1
        s_start, ns, base = offset_s[i], n_sites[i], offset_y[i] + 2
        R, P0, Ei = y[idx_R], y[idx_P0], E_i[i]

        dy[idx_R] = calculate_synthesis_rate(A_i[i], tf_scale, TF_inputs[i]) - B_i[i] * R
        f_folded = calculate_folded_fraction(T, Tm_i[i], c_fold)
        Di_therm = D_i[i] * (1.0 + k_unfold * (1.0 - f_folded))
        P0_act = P0 * f_folded

        k0, P1 = S_all[s_start + 0], y[base + 0]
        dy[idx_P0] = C_i[i] * R - Di_therm * P0 - k0 * P0_act + Ei * P1

        # Simplified for exactly 2 sites
        k1, P2 = S_all[s_start + 1], y[base + 1]
        P1_act = P1 * f_folded
        Dp1_therm = Dp_i[s_start + 0] * (1.0 + k_unfold * (1.0 - f_folded))
        Dp2_therm = Dp_i[s_start + 1] * (1.0 + k_unfold * (1.0 - f_folded))

        dy[base + 0] = k0 * P0_act + Ei * P2 - (k1 + Ei + Dp1_therm + Di_therm) * P1
        dy[base + 1] = k1 * P1_act - (Ei + Dp2_therm + Di_therm) * P2


@njit(fastmath=True, nogil=True)
def combinatorial_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_cache,
                      offset_y, offset_s, n_sites, n_states, trans_from, trans_to, trans_site, trans_off, trans_n,
                      T, Tm_i, c_fold=0.8, k_unfold=4.0):
    for i in range(A_i.shape[0]):
        idx_R, idx_P0 = offset_y[i], offset_y[i] + 1
        s_start, ns, base = offset_s[i], n_sites[i], offset_y[i] + 1
        R, Ei = y[idx_R], E_i[i]

        dy[idx_R] = calculate_synthesis_rate(A_i[i], tf_scale, TF_inputs[i]) - B_i[i] * R
        f_folded = calculate_folded_fraction(T, Tm_i[i], c_fold)
        Di_therm = D_i[i] * (1.0 + k_unfold * (1.0 - f_folded))

        dy[idx_P0] += C_i[i] * R
        dy[base] -= Di_therm * y[base]

        for m in range(1, n_states[i]):
            Pm = y[base + m]
            if Pm == 0.0: continue
            mm, dp_rate = m, 0.0
            while mm != 0:
                lsb = mm & -mm
                mm -= lsb
                j, to = _bit_index_from_lsb(lsb), m ^ lsb
                flux = Ei * Pm
                dy[base + m] -= flux
                dy[base + to] += flux
                dp_rate += Dp_i[s_start + j] * (1.0 + k_unfold * (1.0 - f_folded)) + Di_therm
            dy[base + m] -= dp_rate * Pm

        off = trans_off[i]
        for k in range(trans_n[i]):
            frm, to, j = trans_from[off + k], trans_to[off + k], trans_site[off + k]
            flux = S_cache[s_start + j, 0] * (y[base + frm] * f_folded)
            dy[base + frm] -= flux
            dy[base + to] += flux


# =============================================================================
# 3. Network Parameter Setup (Realistic Arrays)
# =============================================================================
N_PROTS = 3
SITES_PER_PROT = 3
TOTAL_SITES = N_PROTS * SITES_PER_PROT

# --- Base Parameters (Length = N_PROTS) ---
A_i = np.full(N_PROTS, 5.0, dtype=np.float64)
B_i = np.full(N_PROTS, 1.0, dtype=np.float64)
C_i = np.full(N_PROTS, 2.0, dtype=np.float64)
D_i = np.full(N_PROTS, 0.5, dtype=np.float64)
E_i = np.full(N_PROTS, 1.0, dtype=np.float64)
TF_inputs = np.full(N_PROTS, 1.0, dtype=np.float64)
tf_scale = 2.0

# Let's give them different melting points: 38°C, 40°C, and 42°C
Tm_i = np.array([38.0, 40.0, 42.0], dtype=np.float64)

# --- Site Parameters (Length = TOTAL_SITES) ---
Dp_i = np.full(TOTAL_SITES, 0.1, dtype=np.float64)
S_all = np.full(TOTAL_SITES, 3.0, dtype=np.float64)

# --- Memory Offsets ---
n_sites = np.full(N_PROTS, SITES_PER_PROT, dtype=np.int32)
# offset_s defines where each protein's sites start in the S_all/Dp_i arrays
offset_s = np.array([0, 3, 6], dtype=np.int32)

# Distributive/Sequential States: 1 mRNA + 1 Unphos + 3 Mono-Phos = 5 states/protein
offset_y_dist = np.array([0, 5, 10], dtype=np.int32)
total_y_dist = 15

# Combinatorial States: 1 mRNA + 2^3 (8) states = 9 states/protein
n_states = np.full(N_PROTS, 2 ** SITES_PER_PROT, dtype=np.int32)
offset_y_comb = np.array([0, 9, 18], dtype=np.int32)
total_y_comb = 27

# --- Combinatorial Graph Builder ---
# Rebuilding the sparse graph logic from your original codebase
trans_from, trans_to, trans_site = [], [], []
trans_off = np.zeros(N_PROTS, dtype=np.int32)
trans_n = np.zeros(N_PROTS, dtype=np.int32)

cur = 0
for i in range(N_PROTS):
    ns = n_sites[i]
    trans_off[i] = cur
    n_states_i = 1 << ns  # 2^ns

    for m in range(n_states_i):
        for j in range(ns):
            # If bit j is NOT set in state m, add an edge to state (m | 1<<j)
            if (m & (1 << j)) == 0:
                trans_from.append(m)
                trans_to.append(m | (1 << j))
                trans_site.append(j)

    n_i = len(trans_from) - cur
    trans_n[i] = n_i
    cur += n_i

trans_from = np.array(trans_from, dtype=np.int32)
trans_to = np.array(trans_to, dtype=np.int32)
trans_site = np.array(trans_site, dtype=np.int32)
S_cache = np.full((TOTAL_SITES, 1), 3.0, dtype=np.float64)


# =============================================================================
# 4. Updated ODE Wrapper Functions
# =============================================================================
# # Notice we now explicitly pass offset_y_dist or offset_y_comb
# def wrap_distributive(t, y, T):
#     dy = np.zeros_like(y)
#     distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
#                      offset_y_dist, offset_s, n_sites, T, Tm_i)
#     return dy
#
#
# def wrap_sequential(t, y, T):
#     dy = np.zeros_like(y)
#     sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
#                    offset_y_dist, offset_s, n_sites, T, Tm_i)
#     return dy
#
#
# def wrap_combinatorial(t, y, T):
#     dy = np.zeros_like(y)
#     combinatorial_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_cache,
#                       offset_y_comb, offset_s, n_sites, n_states, trans_from, trans_to, trans_site, trans_off, trans_n,
#                       T, Tm_i)
#     return dy

# =============================================================================
# 4. Updated ODE Wrapper Functions (With Dynamic Feedback Loop)
# =============================================================================

def wrap_distributive(t, y, T):
    dy = np.zeros_like(y)

    # --- MINIMAL CHANGE: NETWORK FEEDBACK ---
    # Protein 1's last phosphorylated state activates Protein 2.
    # We dynamically grab the state right before Protein 2's memory block starts.
    prot1_signal = y[offset_y_dist[1] - 1]
    TF_inputs[1] = prot1_signal

    distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                     offset_y_dist, offset_s, n_sites, T, Tm_i)
    return dy


def wrap_sequential(t, y, T):
    dy = np.zeros_like(y)

    # --- MINIMAL CHANGE: NETWORK FEEDBACK ---
    # Protein 1's fully phosphorylated state (P_3) activates Protein 2.
    prot1_signal = y[offset_y_dist[1] - 1]
    TF_inputs[1] = prot1_signal

    sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                   offset_y_dist, offset_s, n_sites, T, Tm_i)
    return dy


def wrap_combinatorial(t, y, T):
    dy = np.zeros_like(y)

    # --- MINIMAL CHANGE: NETWORK FEEDBACK ---
    # Protein 1's fully phosphorylated state (P_111) activates Protein 2.
    # offset_y_comb[1] is 9. So y[8] is the P_111 state of Protein 1.
    prot1_signal = y[offset_y_comb[1] - 1]
    TF_inputs[1] = prot1_signal

    combinatorial_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_cache,
                      offset_y_comb, offset_s, n_sites, n_states, trans_from, trans_to, trans_site, trans_off, trans_n,
                      T, Tm_i)
    return dy
# =============================================================================
# 5. Comprehensive Run and Plot Setup (All States, All Proteins)
# =============================================================================
# Define your new time and resolution parameters
FORWARD_TIME = 10.0   # Increase this to simulate further into the future (e.g., 50, 100, 500)
NUM_POINTS = 10000     # Increase this to get a much smoother, high-resolution curve

t_span = (0, FORWARD_TIME)
t_eval = np.linspace(0, FORWARD_TIME, NUM_POINTS)

temperatures = [
    ("Standard (100% Folded Baseline)", 20.0),
    ("Thermal Model @ 37°C", 37.0),
    ("Thermal Model @ 42°C (Heat Shock)", 42.0)
]


# --- Label Generators ---
def get_dist_labels():
    return ['RNA', 'P_unphos', 'P_site0', 'P_site1', 'P_site2']


def get_seq_labels():
    return ['RNA', 'P_unphos', 'P_1', 'P_2', 'P_3']


def get_comb_labels(n_sites):
    labels = ['RNA']
    for m in range(1 << n_sites):
        # Creates binary strings like P_000, P_001, etc.
        labels.append(f"P_{m:0{n_sites}b}")
    return labels


# --- Define the Models to Plot ---
models = [
    ("Distributive", wrap_distributive, total_y_dist, offset_y_dist, get_dist_labels()),
    ("Sequential", wrap_sequential, total_y_dist, offset_y_dist, get_seq_labels()),
    ("Combinatorial", wrap_combinatorial, total_y_comb, offset_y_comb, get_comb_labels(SITES_PER_PROT))
]

# Loop through each model and create a separate 3x3 figure
for model_name, ode_func, size, offsets, labels in models:
    fig, axes = plt.subplots(N_PROTS, 3, figsize=(18, 12), sharex=True)
    fig.suptitle(f"{model_name} Model Dynamics across Temperatures", fontsize=18, y=0.98)

    y0 = np.zeros(size)

    # Run simulations for the 3 temperature conditions
    solutions = [
        solve_ivp(lambda t, y: ode_func(t, y, temps[1]), t_span, y0, t_eval=t_eval)
        for temps in temperatures
    ]

    # Loop over proteins (Rows)
    for p_idx in range(N_PROTS):
        start_idx = offsets[p_idx]

        # Determine end index based on whether it's the last protein
        if p_idx == N_PROTS - 1:
            end_idx = size
        else:
            end_idx = offsets[p_idx + 1]

        n_states_for_prot = end_idx - start_idx

        # Loop over temperature conditions (Columns)
        for t_idx, sol in enumerate(solutions):
            ax = axes[p_idx, t_idx]

            # Plot every state for this specific protein
            for s_idx in range(n_states_for_prot):
                global_idx = start_idx + s_idx
                ax.plot(sol.t, sol.y[global_idx], linewidth=2, label=labels[s_idx])

            # Formatting
            if p_idx == 0:
                ax.set_title(temperatures[t_idx][0], fontsize=14)
            if t_idx == 0:
                ax.set_ylabel(f"Protein {p_idx + 1}\n(Tm={Tm_i[p_idx]}°C)\nConcentration", fontsize=12)
            if p_idx == N_PROTS - 1:
                ax.set_xlabel('Time', fontsize=12)

            ax.grid(True, linestyle=':', alpha=0.7)

            # Only put the legend in the right-most column to save space
            if t_idx == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Leave room for suptitle and external legend
    plt.show()