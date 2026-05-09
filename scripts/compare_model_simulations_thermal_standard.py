import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from numba import njit


# =============================================================================
# 1. Helper Functions & Numba Kernels (Unchanged)
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


# --- Distributive Kernel ---
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


# --- Sequential Kernel ---
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

        k1, P2 = S_all[s_start + 1], y[base + 1]
        k2, P3 = S_all[s_start + 2], y[base + 2]  # Expanded to 3 sites

        P1_act, P2_act = P1 * f_folded, P2 * f_folded
        Dp1_therm = Dp_i[s_start + 0] * (1.0 + k_unfold * (1.0 - f_folded))
        Dp2_therm = Dp_i[s_start + 1] * (1.0 + k_unfold * (1.0 - f_folded))
        Dp3_therm = Dp_i[s_start + 2] * (1.0 + k_unfold * (1.0 - f_folded))

        dy[base + 0] = k0 * P0_act + Ei * P2 - (k1 + Ei + Dp1_therm + Di_therm) * P1
        dy[base + 1] = k1 * P1_act + Ei * P3 - (k2 + Ei + Dp2_therm + Di_therm) * P2
        dy[base + 2] = k2 * P2_act - (Ei + Dp3_therm + Di_therm) * P3


# --- Combinatorial Kernel ---
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
# 2. Time Series & Network Dummy Data
# =============================================================================
N_PROTS = 3
SITES_PER_PROT = 3
TOTAL_SITES = N_PROTS * SITES_PER_PROT

# Provided Time Series Points
time_points = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])

# Generate dummy External Kinase and TF activities (between 0 and 2)
# We use scipy.interpolate to create continuous functions for the ODE solver
dummy_ext_kinase_data = np.random.uniform(0, 2, len(time_points))
dummy_ext_tf_data = np.random.uniform(0, 2, len(time_points))

interp_ext_kinase = interp1d(time_points, dummy_ext_kinase_data, kind='cubic', fill_value="extrapolate")
interp_ext_tf = interp1d(time_points, dummy_ext_tf_data, kind='cubic', fill_value="extrapolate")

# =============================================================================
# 3. Optimized Alpha / Beta Weights (Dummy Inferred Parameters)
# =============================================================================
# TF Weights (Alpha: Gene Target affinity | Beta: TF Activity weight)
alpha_tf = np.array([1.0, 0.8, 0.5])  # Effect on mRNA Synthesis
beta_tf = np.array([1.0, 1.2, 0.9])

# Kinase Weights (Alpha: Psite affinity | Beta: Kinase Activity weight)
# Size = TOTAL_SITES (9 sites). Sum of alpha per gene usually = 1.
alpha_kin = np.array([
    0.3, 0.3, 0.4,  # Prot 0 sites
    0.5, 0.5, 0.0,  # Prot 1 sites (site 3 is immune to kinase!)
    0.2, 0.6, 0.2  # Prot 2 sites
])
beta_kin = np.array([1.5, 0.8, 1.1])

# Base Parameters
A_i = np.full(N_PROTS, 2.0, dtype=np.float64)
B_i = np.full(N_PROTS, 0.5, dtype=np.float64)
C_i = np.full(N_PROTS, 1.0, dtype=np.float64)
D_i = np.full(N_PROTS, 0.2, dtype=np.float64)
E_i = np.full(N_PROTS, 0.5, dtype=np.float64)
Dp_i = np.full(TOTAL_SITES, 0.1, dtype=np.float64)
Tm_i = np.array([38.0, 40.0, 42.0], dtype=np.float64)

tf_scale = 5.0  # Max fold change from TF activation

# Graph Offsets
n_sites = np.full(N_PROTS, SITES_PER_PROT, dtype=np.int32)
offset_s = np.array([0, 3, 6], dtype=np.int32)
offset_y_dist = np.array([0, 5, 10], dtype=np.int32)
offset_y_comb = np.array([0, 9, 18], dtype=np.int32)
total_y_dist = 15
total_y_comb = 27

# Combinatorial Builder
n_states = np.full(N_PROTS, 2 ** SITES_PER_PROT, dtype=np.int32)
trans_from, trans_to, trans_site = [], [], []
trans_off, trans_n = np.zeros(N_PROTS, dtype=np.int32), np.zeros(N_PROTS, dtype=np.int32)
cur = 0
for i in range(N_PROTS):
    ns = n_sites[i]
    trans_off[i] = cur
    for m in range(1 << ns):
        for j in range(ns):
            if (m & (1 << j)) == 0:
                trans_from.append(m);
                trans_to.append(m | (1 << j));
                trans_site.append(j)
    trans_n[i] = len(trans_from) - cur
    cur += trans_n[i]

trans_from, trans_to, trans_site = np.array(trans_from, dtype=np.int32), np.array(trans_to, dtype=np.int32), np.array(
    trans_site, dtype=np.int32)

# Dynamic arrays to be modified during runtime
TF_inputs = np.zeros(N_PROTS, dtype=np.float64)
S_all = np.zeros(TOTAL_SITES, dtype=np.float64)
S_cache = np.zeros((TOTAL_SITES, 1), dtype=np.float64)


# =============================================================================
# 4. Dynamic Network Wrappers (The Optimization Application)
# =============================================================================
def apply_network_weights(t, y, offset_array):
    """
    Dynamically applies Alpha & Beta weights to calculate instantaneous
    Synthesis (A) and Phosphorylation (S) rates based on the network structure.
    """
    # 1. Get current states
    ext_tf = max(0, interp_ext_tf(t))  # External TF signal
    ext_kin = max(0, interp_ext_kinase(t))  # External Kinase signal

    # Internal fully-phosphorylated states act as TFs/Kinases for downstream targets
    p0_signal = y[offset_array[1] - 1]
    p1_signal = y[offset_array[2] - 1]

    # 2. Update TF_inputs (Synthesis rates via Alpha * Beta * [TF])
    # Prot 0 is driven by external dummy TF
    TF_inputs[0] = alpha_tf[0] * beta_tf[0] * ext_tf
    # Prot 1 is driven by fully-phosphorylated Prot 0
    TF_inputs[1] = alpha_tf[1] * beta_tf[1] * p0_signal
    # Prot 2 is driven by fully-phosphorylated Prot 1
    TF_inputs[2] = alpha_tf[2] * beta_tf[2] * p1_signal

    # 3. Update S_all (Phosphorylation rates via Alpha * Beta * [Kinase])
    # Prot 0 sites driven by external dummy Kinase
    for j in range(3):
        S_all[j] = alpha_kin[j] * beta_kin[0] * ext_kin

    # Prot 1 sites driven by Prot 0 signal
    for j in range(3, 6):
        S_all[j] = alpha_kin[j] * beta_kin[1] * p0_signal

    # Prot 2 sites driven by Prot 1 signal
    for j in range(6, 9):
        S_all[j] = alpha_kin[j] * beta_kin[2] * p1_signal

    # Mirror to S_cache for the Combinatorial kernel
    for j in range(TOTAL_SITES):
        S_cache[j, 0] = S_all[j]


def wrap_distributive(t, y, T):
    dy = np.zeros_like(y)
    apply_network_weights(t, y, offset_y_dist)  # Dynamic update
    distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all, offset_y_dist, offset_s, n_sites,
                     T, Tm_i)
    return dy


def wrap_sequential(t, y, T):
    dy = np.zeros_like(y)
    apply_network_weights(t, y, offset_y_dist)  # Dynamic update
    sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all, offset_y_dist, offset_s, n_sites,
                   T, Tm_i)
    return dy


def wrap_combinatorial(t, y, T):
    dy = np.zeros_like(y)
    apply_network_weights(t, y, offset_y_comb)  # Dynamic update
    combinatorial_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_cache, offset_y_comb, offset_s,
                      n_sites, n_states, trans_from, trans_to, trans_site, trans_off, trans_n, T, Tm_i)
    return dy


# =============================================================================
# 5. Run and Plot
# =============================================================================
t_span = (0, 960.0)  # Full time series
t_eval = np.linspace(0, 960.0, 1000)

temperatures = [
    ("Standard (100% Folded)", 20.0),
    ("Thermal Model @ 37°C", 37.0),
    ("Heat Shock @ 42°C", 42.0)
]


def get_dist_labels(): return ['RNA', 'P_unphos', 'P_site0', 'P_site1', 'P_site2']


def get_seq_labels(): return ['RNA', 'P_unphos', 'P_1', 'P_2', 'P_3']


def get_comb_labels(n): return ['RNA'] + [f"P_{m:0{n}b}" for m in range(1 << n)]


models = [
    ("Distributive", wrap_distributive, total_y_dist, offset_y_dist, get_dist_labels()),
    ("Sequential", wrap_sequential, total_y_dist, offset_y_dist, get_seq_labels()),
    ("Combinatorial", wrap_combinatorial, total_y_comb, offset_y_comb, get_comb_labels(SITES_PER_PROT))
]

for model_name, ode_func, size, offsets, labels in models:
    fig, axes = plt.subplots(N_PROTS, 3, figsize=(18, 12), sharex=True)
    fig.suptitle(f"{model_name} Model Dynamics (Time = 0 to 960)", fontsize=18, y=0.98)

    y0 = np.zeros(size)
    solutions = [solve_ivp(lambda t, y: ode_func(t, y, temps[1]), t_span, y0, t_eval=t_eval) for temps in temperatures]

    for p_idx in range(N_PROTS):
        start_idx = offsets[p_idx]
        end_idx = size if p_idx == N_PROTS - 1 else offsets[p_idx + 1]

        for t_idx, sol in enumerate(solutions):
            ax = axes[p_idx, t_idx]
            for s_idx in range(end_idx - start_idx):
                ax.plot(sol.t, sol.y[start_idx + s_idx], linewidth=2, label=labels[s_idx])

            if p_idx == 0: ax.set_title(temperatures[t_idx][0], fontsize=14)
            if t_idx == 0: ax.set_ylabel(f"Protein {p_idx + 1}\n(Tm={Tm_i[p_idx]}°C)\nConc.", fontsize=12)
            if p_idx == N_PROTS - 1: ax.set_xlabel('Time', fontsize=12)

            ax.grid(True, linestyle=':', alpha=0.7)
            if t_idx == 2: ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()