from tfopt.local.optcon.construct import build_fixed_arrays
from tfopt.local.utils.iodata import load_regulation, load_expression_data, load_tf_protein_data

def load_and_filter_data():
    """
    Load and filter data for the optimization problem.
    This function loads gene expression data, transcription factor (TF) data,
    and regulation data. It filters genes to only include those with at least
    one regulator and filters regulators to only include those present in the
    TF data. The function returns the filtered gene IDs, expression matrix,
    TF IDs, TF protein data, TF phosphorylation site data, TF phosphorylation
    site labels, and the regulation map.
    It raises a ValueError if no genes with regulators are found.
    The function also filters the TF data to only include those present in the
    regulation map.

    The function returns:
    - gene_ids: List of filtered gene IDs.
    - expr_matrix: Filtered expression matrix.
    - expr_time_cols: Time columns for gene expression data.
    - tf_ids: Filtered TF IDs.
    - tf_protein: Dictionary mapping TF IDs to their protein data.
    - tf_psite_data: Dictionary mapping TF IDs to their phosphorylation site data.
    - tf_psite_labels: Dictionary mapping TF IDs to their phosphorylation site labels.
    - tf_time_cols: Time columns for TF data.
    - reg_map: Regulation map, mapping gene IDs to their regulators.
    """
    gene_ids, expr_matrix, expr_time_cols = load_expression_data()
    tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols = load_tf_protein_data()
    reg_map = load_regulation()

    # Filter genes: only keep those with at least one regulator.
    filtered_indices = [i for i, gene in enumerate(gene_ids) if gene in reg_map]
    if len(filtered_indices) == 0:
        raise ValueError("No genes with regulators found. Exiting.")
    gene_ids = [gene_ids[i] for i in filtered_indices]
    expr_matrix = expr_matrix[filtered_indices, :]

    # For each gene, filter regulators to those present in tf_ids.
    relevant_tfs = set()
    for gene in gene_ids:
        regs = reg_map.get(gene, [])
        reg_map[gene] = regs
        relevant_tfs.update(regs)

    # Filter TFs.
    tf_ids_filtered = [tf for tf in tf_ids if tf in relevant_tfs]
    tf_ids = tf_ids_filtered
    tf_protein = {tf: tf_protein[tf] for tf in tf_ids}
    tf_psite_data = {tf: tf_psite_data[tf] for tf in tf_ids}
    tf_psite_labels = {tf: tf_psite_labels[tf] for tf in tf_ids}

    return gene_ids, expr_matrix, expr_time_cols, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols, reg_map


def prepare_data(gene_ids, expr_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols, reg_map):
    """
    Prepares the data for optimization by filtering the expression matrix
    to match the number of time points and building fixed arrays.

    Args:
        gene_ids (list): List of gene IDs.
        expr_matrix (np.ndarray): Gene expression matrix.
        tf_ids (list): List of transcription factor IDs.
        tf_protein (dict): Dictionary mapping TF IDs to their protein data.
        tf_psite_data (dict): Dictionary mapping TF IDs to their phosphorylation site data.
        tf_psite_labels (dict): Dictionary mapping TF IDs to their phosphorylation site labels.
        tf_time_cols (list): Time columns for TF data.
        reg_map (dict): Regulation map, mapping gene IDs to their regulators.
    Returns:
        fixed_arrays (tuple): Tuple containing the fixed arrays:
            - expression_matrix: array of shape (n_genes, T)
            - regulators: array of shape (n_genes, n_reg) with indices into tf_ids.
            - tf_protein_matrix: array of shape (n_TF, T)
            - psite_tensor: array of shape (n_TF, n_psite_max, T), padded with zeros.
            - n_reg: maximum number of regulators per gene.
            - n_psite_max: maximum number of PSites among TFs.
            - psite_labels_arr: list (length n_TF) of lists of PSite names (padded with empty strings).
            - num_psites: array of length n_TF with the actual number of PSites for each TF.
        T_use (int): Number of time points used in the expression matrix.
    """
    T_use = min(expr_matrix.shape[1], len(tf_time_cols))
    expr_matrix = expr_matrix[:, :T_use]
    fixed_arrays = build_fixed_arrays(gene_ids, expr_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map)
    return fixed_arrays, T_use
