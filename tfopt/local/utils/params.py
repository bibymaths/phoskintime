import numpy as np
from tfopt.local.optcon.construct import build_linear_constraints
from tfopt.local.config.logconf import setup_logger 
  
logger = setup_logger()

def get_optimization_parameters(expression_matrix, tf_protein_matrix, n_reg, T_use,
                                psite_labels_arr, num_psites, lb, ub):
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

    bounds_alpha = [(0.0, 1.0)] * n_alpha
    bounds_beta = [(lb, ub)] * len(x0_beta)
    bounds = bounds_alpha + bounds_beta

    lin_cons = build_linear_constraints(n_genes, n_TF, n_reg, n_alpha, beta_start_indices, num_psites, no_psite_tf)

    return x0, n_alpha, beta_start_indices, bounds, no_psite_tf, n_genes, n_TF, num_psites, lin_cons, T_use

def postprocess_results(result, n_alpha, n_genes, n_reg, beta_start_indices, num_psites, reg_map, gene_ids, tf_ids,
                        psite_labels_arr):
    final_x = result.x
    final_alpha = final_x[:n_alpha].reshape((n_genes, n_reg))
    final_beta = []
    for i in range(len(tf_ids)):
        start = beta_start_indices[i]
        length = 1 + num_psites[i]
        final_beta.append(final_x[n_alpha + start: n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)

    # Build and logger.info α mapping.
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
        logger.info(f"   mRNA {tf}: {beta_vec[0]:.4f}")
        for q in range(1, len(beta_vec)):
            label = psite_labels_arr[idx][q-1]
            if label == "":
                label = f"PSite{q}"
            logger.info(f"   {label}: {beta_vec[q]:.4f}")

    # Build and logger.info β mapping.
    # beta_mapping = {}
    # for idx, tf in enumerate(tf_ids):
    #     beta_mapping[tf] = {}
    #     beta_vec = final_beta[idx]
    #     logger.info(f"{tf}:")
    #     beta_mapping[tf][f"mRNA {tf}"] = beta_vec[0]
    #     for q in range(1, len(beta_vec)):
    #         label = psite_labels_arr[idx][q - 1]
    #         if label == "":
    #             label = f"PSite{q}"
    #         beta_mapping[tf][label] = beta_vec[q]
    # logger.info("Mapping of phosphorylation sites to TFs (β parameters):")
    # for tf, mapping in beta_mapping.items():
    #     logger.info(f"{tf}:")
    #     for label, b_val in mapping.items():
    #         logger.info(f"   {label}: {b_val:.4f}")
    return final_x, final_alpha, final_beta