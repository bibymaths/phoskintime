import numpy as np
from tfopt.local.optcon.construct import build_linear_constraints, constraint_alpha_func, constraint_beta_func
from tfopt.local.config.logconf import setup_logger

logger = setup_logger()


def get_optimization_parameters(expression_matrix, tf_protein_matrix, n_reg, T_use,
                                psite_labels_arr, num_psites, lb, ub):
    """
    Prepare the optimization parameters for the optimization problem.

    Args:
        expression_matrix (np.ndarray): Gene expression matrix.
        tf_protein_matrix (np.ndarray): TF protein matrix.
        n_reg (int): Number of regulators.
        T_use (int): Number of time points to use.
        psite_labels_arr (list): List of phosphorylation site labels for each TF.
        num_psites (np.ndarray): Array containing the number of phosphorylation sites for each TF.
        lb (float): Lower bound for beta parameters.
        ub (float): Upper bound for beta parameters.
    Returns:
        x0 (np.ndarray): Initial guess for the optimization variables.
        n_alpha (int): Number of alpha parameters.
        beta_start_indices (np.ndarray): Starting indices for beta parameters.
        bounds (list): List of bounds for the optimization variables.
        no_psite_tf (np.ndarray): Array indicating whether each TF has no phosphorylation sites.
        n_genes (int): Number of genes.
        n_TF (int): Number of transcription factors.

    """
    n_genes = expression_matrix.shape[0]
    n_TF = tf_protein_matrix.shape[0]
    no_psite_tf = np.array([(num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                            for i in range(n_TF)])
    beta_start_indices = np.zeros(n_TF, dtype=np.int32)
    cum = 0
    for i in range(n_TF):
        beta_start_indices[i] = cum
        cum += 1 + num_psites[i]
    n_alpha = n_genes * n_reg
    # x0_alpha = np.full(n_alpha, 1.0 / n_reg)
    # Initialize x0_alpha using uniform random numbers and normalize per gene.
    x0_alpha = np.empty(n_alpha)
    for i in range(n_genes):
        # Sample n_reg values uniformly from [0,1)
        a = np.random.rand(n_reg)
        a /= a.sum()  # Normalize so that the regulators for gene i sum to 1.
        x0_alpha[i * n_reg:(i + 1) * n_reg] = a
    # Initialize x0_beta by sampling uniformly from [lb, ub] and normalizing per TF.
    x0_beta_list = []
    for i in range(n_TF):
        length = 1 + num_psites[i]
        if no_psite_tf[i]:
            # For TFs without any PSite, the beta vector has one element.
            # The constraint forces that element to be 1.0, so we simply use 1.0.
            x0_beta_list.extend([1.0])
        else:
            sample = np.random.uniform(lb, ub, size=length)
            sample /= sample.sum()  # Normalize so that this beta vector sums to 1.
            x0_beta_list.extend(sample.tolist())
            # x0_beta_list.extend([1.0 / length] * length)

    x0_beta = np.array(x0_beta_list)
    x0 = np.concatenate([x0_alpha, x0_beta])

    bounds_alpha = [(0.0, 1.0)] * n_alpha
    bounds_beta = [(lb, ub)] * len(x0_beta)
    bounds = bounds_alpha + bounds_beta

    lin_cons = build_linear_constraints(n_genes, n_TF, n_reg, n_alpha, beta_start_indices, num_psites, no_psite_tf)

    return x0, n_alpha, beta_start_indices, bounds, no_psite_tf, n_genes, n_TF, num_psites, lin_cons, T_use


def postprocess_results(result, n_alpha, n_genes, n_reg, beta_start_indices, num_psites, reg_map, gene_ids, tf_ids,
                        psite_labels_arr):
    """
    Post-process the optimization results to extract the final alpha and beta parameters.

    Args:
        result (OptimizeResult): The result of the optimization.
        n_alpha (int): Number of alpha parameters.
        n_genes (int): Number of genes.
        n_reg (int): Number of regulators.
        beta_start_indices (np.ndarray): Starting indices for beta parameters.
        num_psites (np.ndarray): Array containing the number of phosphorylation sites for each TF.
        reg_map (dict): Regulation map, mapping gene IDs to their regulators.
        gene_ids (list): List of gene IDs.
        tf_ids (list): List of transcription factor IDs.
        psite_labels_arr (list): List of lists containing phosphorylation site labels.

    Returns:
        final_x (np.ndarray): Final optimization result.
        final_alpha (np.ndarray): Final alpha parameters reshaped into a matrix.
        final_beta (np.ndarray): Final beta parameters reshaped into a matrix.
    """
    final_x = result.x
    final_alpha = final_x[:n_alpha].reshape((n_genes, n_reg))
    final_beta = []
    for i in range(len(tf_ids)):
        start = beta_start_indices[i]
        length = 1 + num_psites[i]
        final_beta.append(final_x[n_alpha + start: n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)

    alpha_mapping = {}
    for i, gene in enumerate(gene_ids):
        actual_tfs = [tf for tf in reg_map[gene] if tf in tf_ids]
        alpha_mapping[gene] = {}
        for j, tf in enumerate(actual_tfs):
            alpha_mapping[gene][tf] = final_alpha[i, j]

    logger.info("Mapping of TFs to mRNAs (α values):")
    for gene, mapping in alpha_mapping.items():
        logger.info(f"mRNA {gene}:")
        for tf, a_val in mapping.items():
            logger.info(f"TF   {tf}: {a_val:.4f}")

    logger.info("Mapping of TFs to β parameters:")
    for idx, tf in enumerate(tf_ids):
        beta_vec = final_beta[idx]
        logger.info(f"{tf}:")
        logger.info(f"   TF {tf}: {beta_vec[0]:.4f}")
        for q in range(1, len(beta_vec)):
            label = psite_labels_arr[idx][q - 1]
            if label == "":
                label = f"PSite{q}"
            logger.info(f"   {label}: {beta_vec[q]:.4f}")

    return final_x, final_alpha, final_beta
