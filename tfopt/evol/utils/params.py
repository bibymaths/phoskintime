import subprocess
import numpy as np
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from tfopt.evol.config.logconf import setup_logger
logger = setup_logger()

def create_no_psite_array(n_TF, num_psites, psite_labels_arr):
    """
    Create an array indicating whether each TF has no phosphorylation sites.
    A TF is considered to have no phosphorylation sites if:
    1. The number of phosphorylation sites is zero.
    2. All labels for the phosphorylation sites are empty strings.
    This function is used to determine the initial guess for the beta parameters
    in the optimization process.

    :param n_TF:
    :param num_psites:
    :param psite_labels_arr:
    :return: array of booleans indicating no phosphorylation sites for each TF
    """
    return np.array([(num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                     for i in range(n_TF)])

def compute_beta_indices(num_psites, n_TF):
    """
    Compute the starting indices for the beta parameters for each TF.
    The beta parameters are stored in a flat array, and this function computes
    the starting index for each TF based on the number of phosphorylation sites.
    The starting index for the beta parameters of TF i is given by:
    beta_start_indices[i] = sum(1 + num_psites[j] for j in range(i))
    This function is used to extract the beta parameters from the flat array
    during the optimization process.

    :param num_psites: array of integers indicating the number of phosphorylation sites for each TF
    :param n_TF: number of TFs
    :return: beta_start_indices: array of integers indicating the starting index for each TF
             cum: total number of beta parameters
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
    The initial guess is a flat array containing the alpha and beta parameters.
    The alpha parameters are initialized to 1.0 / n_reg for each mRNA-regulator pair.
    The beta parameters are initialized to 1.0 for the protein and 1.0 / (1 + num_psites[i])
    for each phosphorylation site of TF i.
    The beta parameters for TFs with no phosphorylation sites are initialized to 1.0.
    The initial guess is used as the starting point for the optimization process.
    The initial guess is a flat array of length n_alpha + n_beta_total,
    where n_alpha is the number of alpha parameters and n_beta_total is the total number of beta parameters.
    The alpha parameters are stored in the first n_alpha elements of the array,
    and the beta parameters are stored in the remaining elements.
    The beta parameters are stored in the order of the TFs, with the protein parameter first
    followed by the phosphorylation site parameters.
    The beta parameters for TFs with no phosphorylation sites are stored as a single value.
    The beta parameters for TFs with phosphorylation sites are stored as a vector of length 1 + num_psites[i].

    :param n_mRNA:
    :param n_reg:
    :param n_TF:
    :param num_psites:
    :param no_psite_tf:
    :return: initial guess array, n_alpha
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
    The lower bounds are set to 0 for the alpha parameters and lb for the beta parameters.
    The upper bounds are set to 1 for the alpha parameters and ub for the beta parameters.
    The bounds are used to constrain the optimization process and ensure that the parameters
    are within a reasonable range.
    The bounds are stored in two separate arrays: xl and xu.
    The xl array contains the lower bounds for each parameter,
    and the xu array contains the upper bounds.

    :param n_alpha: number of alpha parameters
    :param n_beta_total: total number of beta parameters
    :param lb: lower bound for beta parameters
    :param ub: upper bound for beta parameters
    :return: xl: lower bounds array
             xu: upper bounds array
    """
    xl = np.concatenate([np.zeros(n_alpha), lb * np.ones(n_beta_total)])
    xu = np.concatenate([np.ones(n_alpha), ub * np.ones(n_beta_total)])
    return xl, xu

def get_parallel_runner():
    """
    Get a parallel runner for multi-threading.
    This function uses the lscpu command to determine the number of threads available
    on the system and creates a ThreadPool with that number of threads.
    The ThreadPool is used to parallelize the optimization process and speed up the computation.

    :return: runner: StarmapParallelization object for parallel execution
                pool: ThreadPool object for managing threads
    """
    n_threads_cmd = "lscpu -p | grep -v '^#' | wc -l"
    n_threads = int(subprocess.check_output(n_threads_cmd, shell=True).decode().strip())
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)
    return runner, pool

def extract_best_solution(res, n_alpha, n_mRNA, n_reg, n_TF, num_psites, beta_start_indices):
    """
    Extract the best solution from the optimization results.
    This function finds the best solution based on the Pareto front,
    which is a set of non-dominated solutions in the objective space.
    The best solution is determined by minimizing the weighted sum of the objectives.
    The alpha parameters are reshaped into a matrix of shape (n_mRNA, n_reg),
    and the beta parameters are extracted based on the starting indices and
    the number of phosphorylation sites for each TF.
    The beta parameters are stored in a list of arrays, where each array corresponds to a TF
    and contains the protein parameter and the phosphorylation site parameters.
    The function returns the final alpha and beta parameters, the best objectives,
    and the final decision variables.

    :param res: optimization results
    :param n_alpha: number of alpha parameters
    :param n_mRNA: number of mRNAs
    :param n_reg: number of regulators
    :param n_TF: number of transcription factors
    :param num_psites: number of phosphorylation sites for each TF
    :param beta_start_indices: starting indices for the beta parameters

    :return: final_alpha: final alpha parameters (n_mRNA x n_reg)
                final_beta: final beta parameters (n_TF x (1 + num_psites))
                best_objectives: best objectives (3 objectives)
                final_x: final decision variables (flat array)
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
        final_beta.append(final_x[n_alpha + start : n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)
    return final_alpha, final_beta, best_objectives, final_x

def print_alpha_mapping(mRNA_ids, reg_map, TF_ids, final_alpha):
    """
    Print the mapping of transcription factors (TFs) to mRNAs with their corresponding alpha values.
    This function iterates through the mRNA IDs and their corresponding regulators,
    and logs the TFs that are present in the final alpha matrix.
    The alpha values are printed for each TF that regulates the mRNA.

    :param mRNA_ids: List of mRNA gene identifiers.
    :param reg_map: Regulation map, mapping mRNA genes to their regulators.
    :param TF_ids: List of transcription factor identifiers.
    :param final_alpha: Final alpha parameters (n_mRNA x n_reg).
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
    This function iterates through the TF IDs and their corresponding beta values,
    and logs the beta values for each TF.
    The beta values are printed for the protein and each phosphorylation site,
    with appropriate labels.

    :param TF_ids: List of transcription factor identifiers.
    :param final_beta: Final beta parameters (n_TF x (1 + num_psites)).
    :param psite_labels_arr: Array of phosphorylation site labels for each TF.
    """
    logger.info("Mapping of TFs to β parameters:")
    for idx, tf in enumerate(TF_ids):
        beta_vec = final_beta[idx]
        logger.info(f"{tf}:")
        logger.info(f"   TF {tf}: {beta_vec[0]:.4f}")
        for q in range(1, len(beta_vec)):
            label = psite_labels_arr[idx][q-1]
            if label == "":
                label = f"PSite{q}"
            logger.info(f"   {label}: {beta_vec[q]:.4f}")
