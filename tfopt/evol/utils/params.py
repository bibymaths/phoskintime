import subprocess
import numpy as np
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from tfopt.evol.config.logconf import setup_logger

logger = setup_logger()


def create_no_psite_array(n_TF, num_psites, psite_labels_arr):
    """
    Create an array indicating whether each TF has no phosphorylation sites.

    Args:
        n_TF (int): Number of transcription factors.
        num_psites (list): List of number of phosphorylation sites for each TF.
        psite_labels_arr (list): List of phosphorylation site labels for each TF.

    Returns:
        no_psite_tf (np.ndarray): Array indicating whether each TF has no phosphorylation sites.
    """
    return np.array([(num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                     for i in range(n_TF)])


def compute_beta_indices(num_psites, n_TF):
    """
    Compute the starting indices for the beta parameters for each TF.

    Args:
        num_psites (list): List of number of phosphorylation sites for each TF.
        n_TF (int): Number of transcription factors.

    Returns:
        beta_start_indices (np.ndarray): Array of starting indices for the beta parameters.
        cum (int): Total number of beta parameters.
    """
    beta_start_indices = np.zeros(n_TF, dtype=np.int32)
    cum = 0
    for i in range(n_TF):
        beta_start_indices[i] = cum
        cum += 1 + num_psites[i]
    return beta_start_indices, cum


def create_initial_guess(n_mRNA, n_reg, n_TF, num_psites, no_psite_tf):
    """
    Create the initial guess for the optimization variables.

    Args:
        n_mRNA (int): Number of mRNAs.
        n_reg (int): Number of regulators.
        n_TF (int): Number of transcription factors.
        num_psites (list): List of number of phosphorylation sites for each TF.
        no_psite_tf (np.ndarray): Array indicating whether each TF has no phosphorylation sites.

    Returns:
        x0 (np.ndarray): Initial guess for the optimization variables.
        n_alpha (int): Number of alpha parameters.
    """
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
    """
    Create the lower and upper bounds for the optimization variables.

    Args:
        n_alpha (int): Number of alpha parameters.
        n_beta_total (int): Total number of beta parameters.
        lb (float): Lower bound for the optimization variables.
        ub (float): Upper bound for the optimization variables.

    Returns:
        xl (np.ndarray): Lower bounds for the optimization variables.
        xu (np.ndarray): Upper bounds for the optimization variables.
    """
    xl = np.concatenate([np.zeros(n_alpha), lb * np.ones(n_beta_total)])
    xu = np.concatenate([np.ones(n_alpha), ub * np.ones(n_beta_total)])
    return xl, xu


def get_parallel_runner():
    """
    Get a parallel runner for multi-threading.

    Returns:
        runner: Parallelization runner.
        pool: ThreadPool instance for parallel execution.
    """
    n_threads_cmd = "lscpu -p | grep -v '^#' | wc -l"
    n_threads = int(subprocess.check_output(n_threads_cmd, shell=True).decode().strip())
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)
    return runner, pool


def extract_best_solution(res, n_alpha, n_mRNA, n_reg, n_TF, num_psites, beta_start_indices):
    """
    Extract the best solution from the optimization results.

    Args:
        res: Optimization results.
        n_alpha (int): Number of alpha parameters.
        n_mRNA (int): Number of mRNAs.
        n_reg (int): Number of regulators.
        n_TF (int): Number of transcription factors.
        num_psites (list): List of number of phosphorylation sites for each TF.
        beta_start_indices (np.ndarray): Array of starting indices for the beta parameters.

    Returns:
        final_alpha (np.ndarray): Final alpha parameters.
        final_beta (np.ndarray): Final beta parameters.
        best_objectives (np.ndarray): Best objectives from the Pareto front.
        final_x (np.ndarray): Final optimization variables.
    """
    pareto_front = np.array([ind.F for ind in res.pop])
    # Scoring the Pareto front
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
        final_beta.append(final_x[n_alpha + start: n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)
    return final_alpha, final_beta, best_objectives, final_x


def print_alpha_mapping(mRNA_ids, reg_map, TF_ids, final_alpha):
    """
    Print the mapping of transcription factors (TFs) to mRNAs with their corresponding alpha values.

    Args:
        mRNA_ids (list): List of mRNA identifiers.
        reg_map (dict): Mapping of genes to their regulators.
        TF_ids (list): List of TF identifiers.
        final_alpha (np.ndarray): Final alpha parameters (mRNA x TF).
    """
    logger.info("Mapping of TFs to mRNAs (α values):")
    for i, mrna in enumerate(mRNA_ids):
        actual_tfs = [tf for tf in reg_map[mrna] if tf in TF_ids]
        logger.info(f"mRNA {mrna}:")
        for j, tf in enumerate(actual_tfs):
            logger.info(f"TF   {tf}: {final_alpha[i, j]:.4f}")


def print_beta_mapping(TF_ids, final_beta, psite_labels_arr):
    """
    Print the mapping of transcription factors (TFs) to their beta parameters.

    Args:
        TF_ids (list): List of TF identifiers.
        final_beta (np.ndarray): Final beta parameters (TF x β).
        psite_labels_arr (list): List of phosphorylation site labels for each TF.
    """
    logger.info("Mapping of TFs to β parameters:")
    for idx, tf in enumerate(TF_ids):
        beta_vec = final_beta[idx]
        logger.info(f"{tf}:")
        logger.info(f"   TF {tf}: {beta_vec[0]:.4f}")
        for q in range(1, len(beta_vec)):
            label = psite_labels_arr[idx][q - 1]
            if label == "":
                label = f"PSite{q}"
            logger.info(f"   {label}: {beta_vec[q]:.4f}")
