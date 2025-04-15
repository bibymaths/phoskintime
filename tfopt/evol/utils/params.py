import subprocess
from multiprocessing.pool import ThreadPool

import numpy as np
from pymoo.core.problem import StarmapParallelization

from tfopt.evol.config.logconf import setup_logger
  
logger = setup_logger()

# -------------------------------
# Helper Functions for Parameters
# -------------------------------
def create_no_psite_array(n_TF, num_psites, psite_labels_arr):
    return np.array([(num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                     for i in range(n_TF)])

def compute_beta_indices(num_psites, n_TF):
    beta_start_indices = np.zeros(n_TF, dtype=np.int32)
    cum = 0
    for i in range(n_TF):
        beta_start_indices[i] = cum
        cum += 1 + num_psites[i]
    return beta_start_indices, cum  # cum is the total number of beta parameters

def create_initial_guess(n_mRNA, n_reg, n_TF, num_psites, no_psite_tf):
    n_alpha = n_mRNA * n_reg
    x0_alpha = np.full(n_alpha, 1.0 / n_reg)
    x0_beta_list = []
    for i in range(n_TF):
        if no_psite_tf[i]:
            x0_beta_list.extend([1.0])
        else:
            length = 1 + num_psites[i]
            x0_beta_list.extend([1.0 / length] * length)
    x0_beta = np.array(x0_beta_list)
    x0 = np.concatenate([x0_alpha, x0_beta])
    return x0, n_alpha

def create_bounds(n_alpha, n_beta_total, lb, ub):
    xl = np.concatenate([np.zeros(n_alpha), lb * np.ones(n_beta_total)])
    xu = np.concatenate([np.ones(n_alpha), ub * np.ones(n_beta_total)])
    return xl, xu

def get_parallel_runner():
    n_threads_cmd = "lscpu -p | grep -v '^#' | wc -l"
    n_threads = int(subprocess.check_output(n_threads_cmd, shell=True).decode().strip())
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)
    return runner, pool

def extract_best_solution(res, n_alpha, n_mRNA, n_reg, n_TF, num_psites, beta_start_indices):
    pareto_front = np.array([ind.F for ind in res.pop])
    weights = np.array([1.0, 1.0, 1.0])
    scores = pareto_front[:, 0] + weights[1] * np.abs(pareto_front[:, 1]) + weights[2] * np.abs(pareto_front[:, 2])
    best_index = np.argmin(scores)
    best_solution = res.pop[best_index]
    best_objectives = pareto_front[best_index]
    final_x = best_solution.X
    final_alpha = final_x[:n_alpha].reshape((n_mRNA, n_reg))
    final_beta = []
    for i in range(n_TF):
        start = beta_start_indices[i]
        length = 1 + num_psites[i]
        final_beta.append(final_x[n_alpha + start : n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)
    return final_alpha, final_beta, best_objectives, final_x

def print_alpha_mapping(mRNA_ids, reg_map, TF_ids, final_alpha):
    logger.info("Mapping of TFs to mRNAs (α values):")
    for i, mrna in enumerate(mRNA_ids):
        actual_tfs = [tf for tf in reg_map[mrna] if tf in TF_ids]
        logger.info(f"mRNA {mrna}:")
        for j, tf in enumerate(actual_tfs):
            logger.info(f"TF   {tf}: {final_alpha[i, j]:.4f}")

def print_beta_mapping(TF_ids, final_beta, psite_labels_arr):
    logger.info("Mapping of TFs to β parameters:")
    for idx, tf in enumerate(TF_ids):
        beta_vec = final_beta[idx]
        logger.info(f"{tf}:")
        logger.info(f"   mRNA {tf}: {beta_vec[0]:.4f}")
        for q in range(1, len(beta_vec)):
            label = psite_labels_arr[idx][q-1]
            if label == "":
                label = f"PSite{q}"
            logger.info(f"   {label}: {beta_vec[q]:.4f}")
