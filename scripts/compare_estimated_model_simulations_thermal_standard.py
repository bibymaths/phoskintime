import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from numba import njit
import logging
import sys

# pymoo imports
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from multiprocessing.pool import ThreadPool

# =============================================================================
# 0. Logger Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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
        k2, P3 = S_all[s_start + 2], y[base + 2]
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

time_points = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])

dummy_ext_kinase_data = np.random.uniform(0, 2, len(time_points))
dummy_ext_tf_data = np.random.uniform(0, 2, len(time_points))

interp_ext_kinase = interp1d(time_points, dummy_ext_kinase_data, kind='cubic', fill_value="extrapolate")
interp_ext_tf = interp1d(time_points, dummy_ext_tf_data, kind='cubic', fill_value="extrapolate")

# Create TARGET multi-omics dummy data
np.random.seed(42)
target_RNA = np.random.uniform(0.5, 2.0, (N_PROTS, len(time_points)))
target_TotalProt = target_RNA * np.random.uniform(1.0, 3.0, (N_PROTS, len(time_points)))
target_Active = target_TotalProt * np.random.uniform(0.1, 0.5, (N_PROTS, len(time_points)))

# =============================================================================
# 3. Network Structure & Alpha/Beta Weights (Read-Only Globals)
# =============================================================================
alpha_tf = np.array([1.0, 0.8, 0.5])
beta_tf = np.array([1.0, 1.2, 0.9])
alpha_kin = np.array([0.3, 0.3, 0.4, 0.5, 0.5, 0.0, 0.2, 0.6, 0.2])
beta_kin = np.array([1.5, 0.8, 1.1])

Dp_i = np.full(TOTAL_SITES, 0.1, dtype=np.float64)
Tm_i = np.array([38.0, 40.0, 42.0], dtype=np.float64)
tf_scale = 5.0

n_sites = np.full(N_PROTS, SITES_PER_PROT, dtype=np.int32)
offset_s = np.array([0, 3, 6], dtype=np.int32)
offset_y_dist = np.array([0, 5, 10], dtype=np.int32)
offset_y_comb = np.array([0, 9, 18], dtype=np.int32)
total_y_dist = 15
total_y_comb = 27

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
                trans_from.append(m)
                trans_to.append(m | (1 << j))
                trans_site.append(j)
    trans_n[i] = len(trans_from) - cur
    cur += trans_n[i]

trans_from = np.array(trans_from, dtype=np.int32)
trans_to = np.array(trans_to, dtype=np.int32)
trans_site = np.array(trans_site, dtype=np.int32)


def get_dist_labels(): return ['RNA', 'P_unphos', 'P_site0', 'P_site1', 'P_site2']


def get_seq_labels(): return ['RNA', 'P_unphos', 'P_1', 'P_2', 'P_3']


def get_comb_labels(n): return ['RNA'] + [f"P_{m:0{n}b}" for m in range(1 << n)]


# =============================================================================
# 4. Thread-Safe Evaluation Factory (Replaces old wrappers)
# =============================================================================
def simulate_and_evaluate(params, m_name, temp, offsets, size, t_eval_pts, return_sol=False):
    """
    Creates independent memory buffers for every parallel evaluation thread.
    Returns either the 3 Objective Losses [SSE_RNA, SSE_Tot, SSE_Act] or the SciPy sol object.
    """
    # 1. Localize mutable parameters from the optimizer
    loc_A = params[0:3].copy()
    loc_B = params[3:6].copy()
    loc_C = params[6:9].copy()
    loc_D = params[9:12].copy()
    loc_E = params[12:15].copy()

    # 2. Localize network states
    loc_TF_inputs = np.zeros(N_PROTS, dtype=np.float64)
    loc_S_all = np.zeros(TOTAL_SITES, dtype=np.float64)
    loc_S_cache = np.zeros((TOTAL_SITES, 1), dtype=np.float64)

    def local_wrapper(t, y, T):
        ext_tf = max(0, interp_ext_tf(t))
        ext_kin = max(0, interp_ext_kinase(t))

        p0_signal = y[offsets[1] - 1]
        p1_signal = y[offsets[2] - 1]

        loc_TF_inputs[0] = alpha_tf[0] * beta_tf[0] * ext_tf
        loc_TF_inputs[1] = alpha_tf[1] * beta_tf[1] * p0_signal
        loc_TF_inputs[2] = alpha_tf[2] * beta_tf[2] * p1_signal

        for j in range(3): loc_S_all[j] = alpha_kin[j] * beta_kin[0] * ext_kin
        for j in range(3, 6): loc_S_all[j] = alpha_kin[j] * beta_kin[1] * p0_signal
        for j in range(6, 9): loc_S_all[j] = alpha_kin[j] * beta_kin[2] * p1_signal

        for j in range(TOTAL_SITES): loc_S_cache[j, 0] = loc_S_all[j]

        dy = np.zeros_like(y)
        if m_name == "Distributive":
            distributive_rhs(y, dy, loc_A, loc_B, loc_C, loc_D, Dp_i, loc_E, tf_scale, loc_TF_inputs, loc_S_all,
                             offsets, offset_s, n_sites, T, Tm_i)
        elif m_name == "Sequential":
            sequential_rhs(y, dy, loc_A, loc_B, loc_C, loc_D, Dp_i, loc_E, tf_scale, loc_TF_inputs, loc_S_all, offsets,
                           offset_s, n_sites, T, Tm_i)
        elif m_name == "Combinatorial":
            combinatorial_rhs(y, dy, loc_A, loc_B, loc_C, loc_D, Dp_i, loc_E, tf_scale, loc_TF_inputs, loc_S_cache,
                              offsets, offset_s, n_sites, n_states, trans_from, trans_to, trans_site, trans_off,
                              trans_n, T, Tm_i)
        return dy

    y0 = np.zeros(size)
    try:
        sol = solve_ivp(lambda t, y: local_wrapper(t, y, temp),
                        (0, max(t_eval_pts)), y0, t_eval=t_eval_pts, method='LSODA')

        if return_sol:
            return sol

        if not sol.success:
            return [1e6, 1e6, 1e6]

        sse_rna, sse_tot, sse_act = 0.0, 0.0, 0.0
        for p_idx in range(N_PROTS):
            start = offsets[p_idx]
            end = size if p_idx == N_PROTS - 1 else offsets[p_idx + 1]

            sse_rna += np.sum((target_RNA[p_idx] - sol.y[start]) ** 2)
            sse_tot += np.sum((target_TotalProt[p_idx] - np.sum(sol.y[start + 1: end], axis=0)) ** 2)
            sse_act += np.sum((target_Active[p_idx] - sol.y[end - 1]) ** 2)

        return [sse_rna, sse_tot, sse_act]

    except Exception:
        if return_sol: return None
        return [1e6, 1e6, 1e6]


# =============================================================================
# 5. Pymoo Problem Setup & Multi-Objective Optimization
# =============================================================================
class MultiOmicsProblem(Problem):
    def __init__(self, runner, m_name, temp, offsets, size):
        # We define 3 Objectives (F1=RNA, F2=TotalProt, F3=ActiveSignal)
        super().__init__(n_var=15, n_obj=3, n_ieq_constr=0, xl=0.01, xu=10.0)
        self.runner = runner  # This receives pool.starmap
        self.m_name = m_name
        self.temp = temp
        self.offsets = offsets
        self.size = size

    def _evaluate(self, X, out, *args, **kwargs):
        # X is a 2D array of shape (pop_size, 15).
        # 1. Package the arguments for every individual in the population
        tasks = [
            (X[i, :], self.m_name, self.temp, self.offsets, self.size, time_points)
            for i in range(len(X))
        ]

        # 2. Run the evaluations in parallel using the ThreadPool starmap
        res = self.runner(simulate_and_evaluate, tasks)

        # 3. Stack the list of results back into a (pop_size, 3) matrix for pymoo
        out["F"] = np.vstack(res)


class LoggerCallback(Callback):
    def notify(self, algorithm):
        gen = algorithm.n_gen
        if algorithm.pop is not None and len(algorithm.pop) > 0:
            F = algorithm.pop.get("F")
            min_F = F.min(axis=0)
            logger.info(
                f"   Generation {gen:03d} | Best Partial Objs [RNA: {min_F[0]:.2f}, Tot: {min_F[1]:.2f}, Act: {min_F[2]:.2f}]")

model_configs = [
    ("Distributive", total_y_dist, offset_y_dist, get_dist_labels()),
    ("Sequential", total_y_dist, offset_y_dist, get_seq_labels()),
    ("Combinatorial", total_y_comb, offset_y_comb, get_comb_labels(SITES_PER_PROT))
]

temperatures = [("Baseline (20°C)", 20.0), ("Physiological (37°C)", 37.0), ("Heat Shock (42°C)", 42.0)]
optimized_params = {}

# Set up UNSGA3 Parameters
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)  # Creates a pop_size of 45
algo = UNSGA3(ref_dirs=ref_dirs, pop_size=45)
n_generations = 300  # Set low for demonstration. Increase to 50+ for production runs.

logger.info("Starting pymoo UNSGA3 Optimization with ThreadPool starmap...")

# ThreadPool allows parallel Numba (nogil=True) without multiprocess serialization overhead
pool = ThreadPool(processes=48)

for m_name, size, offsets, _ in model_configs:
    for t_name, temp in temperatures:
        logger.info(f"--- Optimizing {m_name} Model at {t_name} ---")

        problem = MultiOmicsProblem(pool.starmap, m_name, temp, offsets, size)

        res = pymoo_minimize(problem, algo, termination=('n_gen', n_generations),
                             callback=LoggerCallback(), seed=1, verbose=False)

        # Select the single solution from the Pareto front that minimizes the sum of all 3 objectives
        if res.F is not None:
            best_idx = np.argmin(res.F.sum(axis=1))
            optimized_params[(m_name, temp)] = res.X[best_idx]
            logger.info(f"Finished {m_name} @ {temp}°C | Selected Aggregated SSE: {res.F[best_idx].sum():.4f}\n")
        else:
            optimized_params[(m_name, temp)] = np.ones(15)

pool.close()
pool.join()
logger.info("All UNSGA3 optimizations complete. Generating Grids...")

# =============================================================================
# 6. Grid Plotting (Per-Model Grids using Optimized Parameters)
# =============================================================================
t_eval_high_res = np.linspace(0, max(time_points), 1000)

for m_name, size, offsets, labels in model_configs:
    fig, axes = plt.subplots(N_PROTS, 3, figsize=(18, 12), sharex=True)
    fig.suptitle(f"{m_name} Model Dynamics", fontsize=18, y=0.98)

    for t_idx, (t_name, temp) in enumerate(temperatures):
        best_p = optimized_params[(m_name, temp)]

        # Run high-res simulation using our thread-safe factory wrapper
        sol = simulate_and_evaluate(best_p, m_name, temp, offsets, size, t_eval_high_res, return_sol=True)

        for p_idx in range(N_PROTS):
            ax = axes[p_idx, t_idx]
            start_idx = offsets[p_idx]
            end_idx = size if p_idx == N_PROTS - 1 else offsets[p_idx + 1]
            n_states_for_prot = end_idx - start_idx

            if sol is not None and sol.success:
                for s_idx in range(n_states_for_prot):
                    global_idx = start_idx + s_idx
                    ax.plot(sol.t, sol.y[global_idx], linewidth=1.5, alpha=0.7, label=labels[s_idx])

                sim_total = np.sum(sol.y[start_idx + 1: end_idx], axis=0)
                ax.plot(sol.t, sim_total, 'k--', linewidth=2, label='Simulated Total Prot')

            # Overlay target data
            ax.scatter(time_points, target_RNA[p_idx], marker='x', color='blue', s=60, zorder=5, label='Target RNA')
            ax.scatter(time_points, target_Active[p_idx], marker='o', color='red', s=40, zorder=5,
                       label='Target Active Signal')
            ax.scatter(time_points, target_TotalProt[p_idx], marker='s', color='black', alpha=0.5, s=40, zorder=5,
                       label='Target Total Prot')

            if p_idx == 0: ax.set_title(t_name, fontsize=14)
            if t_idx == 0: ax.set_ylabel(f"Protein {p_idx + 1}\n(Tm={Tm_i[p_idx]}°C)\nConc.", fontsize=12)
            if p_idx == N_PROTS - 1: ax.set_xlabel('Time', fontsize=12)

            ax.grid(True, linestyle=':', alpha=0.7)
            if t_idx == 2 and p_idx == 0:
                handles, lbls = ax.get_legend_handles_labels()
                by_label = dict(zip(lbls, handles))
                ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show(dpi=300)