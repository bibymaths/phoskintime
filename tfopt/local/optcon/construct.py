import numpy as np
from scipy.optimize import LinearConstraint


def build_fixed_arrays(gene_ids, expression_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map):
    """
    Builds fixed-shape arrays from the input data.

    Args:
        - gene_ids: list of mRNA identifiers.
        - expression_matrix: array of shape (n_genes, T) with mRNA expression levels.
        - tf_ids: list of TF identifiers.
        - tf_protein: dict mapping TFs to their protein levels.
        - tf_psite_data: dict mapping TFs to their phosphorylation sites.
        - tf_psite_labels: dict mapping TFs to their phosphorylation site labels.
        - reg_map: mapping of genes to their regulators (TFs).

    Returns:
        - expression_matrix: array of shape (n_genes, T) with mRNA expression levels.
        - regulators: array of shape (n_genes, n_reg) with TF indices.
        - tf_protein_matrix: array of shape (n_TF, T) with TF protein levels.
        - psite_tensor: array of shape (n_TF, n_psite_max, T) with phosphorylation sites.
        - n_reg: number of regulators.
        - n_psite_max: maximum number of phosphorylation sites across all TFs.
        - psite_labels_arr: list of labels for each TF's phosphorylation sites.
        - num_psites: array indicating the number of phosphorylation sites for each TF.
    """
    n_genes, T = expression_matrix.shape

    # Map TF id to index.
    tf_index = {tf: idx for idx, tf in enumerate(tf_ids)}
    n_TF = len(tf_ids)

    # Determine max number of valid regulators per gene, filtering out TFs not present in tf_ids.
    reg_list = []
    for gene in gene_ids:
        regs = [tf for tf in reg_map.get(gene, []) if tf in tf_ids]
        reg_list.append(regs)
    n_reg = max(len(regs) for regs in reg_list) if reg_list else 1

    # Build regulators array (n_genes x n_reg), padded with -1 to mark invalid.
    regulators = np.full((n_genes, n_reg), -1, dtype=np.int32)
    for i, regs in enumerate(reg_list):
        for j, tf in enumerate(regs):
            regulators[i, j] = tf_index.get(tf, -1)

    tf_protein_matrix = np.zeros((n_TF, T), dtype=np.float64)
    for tf, idx in tf_index.items():
        if tf_protein.get(tf) is not None:
            tf_protein_matrix[idx, :] = tf_protein[tf][:T]
        else:
            tf_protein_matrix[idx, :] = np.zeros(T)

    # For each TF, record the actual number of PSites.
    num_psites = np.zeros(n_TF, dtype=np.int32)
    for i, tf in enumerate(tf_ids):
        num_psites[i] = len(tf_psite_data.get(tf, []))
    # Maximum number of PSites across all TFs.
    n_psite_max = int(np.max(num_psites)) if np.max(num_psites) > 0 else 0

    # Build psite_tensor and psite_labels_arr.
    psite_tensor = np.zeros((n_TF, n_psite_max, T), dtype=np.float64)
    psite_labels_arr = []
    for tf, idx in tf_index.items():
        psites = tf_psite_data.get(tf, [])
        labels = tf_psite_labels.get(tf, [])
        for j in range(n_psite_max):
            if j < len(psites):
                psite_tensor[idx, j, :] = psites[j][:T]
            else:
                psite_tensor[idx, j, :] = np.zeros(T)
        padded_labels = labels + [""] * (n_psite_max - len(labels))
        psite_labels_arr.append(padded_labels)

    return expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, n_psite_max, psite_labels_arr, num_psites


def constraint_alpha_func(x, n_genes, n_reg):
    """
    For each gene, the sum of its alpha parameters must equal 1.

    Args:
        x (np.ndarray): Decision vector.
        n_genes (int): Number of genes.
        n_reg (int): Number of regulators.

    Returns:
        np.ndarray: Array of constraints.
    """
    cons = []
    for i in range(n_genes):
        s = 0.0
        for r in range(n_reg):
            s += x[i * n_reg + r]
        cons.append(s - 1.0)
    return np.array(cons)


def constraint_beta_func(x, n_alpha, n_TF, beta_start_indices, num_psites, no_psite_tf):
    """
    For each TF, the sum of its beta parameters must equal 1.

    Args:
        x (np.ndarray): Decision vector.
        n_alpha (int): Number of alpha parameters.
        n_TF (int): Number of transcription factors.
        beta_start_indices (list): List of starting indices for beta parameters.
        num_psites (list): List of number of phosphorylation sites for each TF.
        no_psite_tf (list): List indicating if a TF has no phosphorylation site.

    Returns:
        np.ndarray: Array of constraints.
    """
    cons = []
    for tf in range(n_TF):
        length = 1 + num_psites[tf]  # Total beta parameters for TF tf.
        start = n_alpha + beta_start_indices[tf]
        beta_vec = x[start: start + length]
        cons.append(np.sum(beta_vec) - 1.0)
        if no_psite_tf[tf]:
            for q in range(1, length):
                cons.append(beta_vec[q])
    return np.array(cons)


def build_linear_constraints(n_genes, n_TF, n_reg, n_alpha, beta_start_indices, num_psites, no_psite_tf):
    """
    Build linear constraints for the transcription factor optimization problem.

    Args:
        n_genes (int): Number of genes.
        n_TF (int): Number of transcription factors.
        n_reg (int): Number of regulators.
        n_alpha (int): Number of alpha parameters.
        beta_start_indices (list): List of starting indices for beta parameters.
        num_psites (list): List of number of phosphorylation sites for each TF.
        no_psite_tf (list): List indicating if a TF has no phosphorylation site.

    Returns:
        list: List of linear constraints.
    """
    total_vars = n_alpha + sum(1 + num_psites[i] for i in range(n_TF))

    # --- Alpha constraints ---
    alpha_constraints_matrix = []
    for i in range(n_genes):
        row = np.zeros(total_vars)
        for j in range(n_reg):
            row[i * n_reg + j] = 1.0
        alpha_constraints_matrix.append(row)
    alpha_constraints_matrix = np.array(alpha_constraints_matrix)
    alpha_constraint = LinearConstraint(alpha_constraints_matrix, lb=1.0, ub=1.0)

    # --- Beta constraints ---
    beta_constraint_rows = []
    lb_list = []
    ub_list = []

    for tf in range(n_TF):
        start = n_alpha + beta_start_indices[tf]
        length = 1 + num_psites[tf]
        row = np.zeros(total_vars)
        row[start: start + length] = 1.0
        beta_constraint_rows.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)

        if no_psite_tf[tf]:
            for q in range(1, length):
                row = np.zeros(total_vars)
                row[start + q] = 1.0
                beta_constraint_rows.append(row)
                lb_list.append(0.0)
                ub_list.append(0.0)

    beta_constraints_matrix = np.array(beta_constraint_rows)
    beta_constraint = LinearConstraint(beta_constraints_matrix, lb=lb_list, ub=ub_list)

    return [alpha_constraint, beta_constraint]
