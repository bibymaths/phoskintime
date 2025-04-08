from tfopt.local.optcon.construct import build_fixed_arrays
from tfopt.local.utils.iodata import load_regulation, load_expression_data, load_tf_protein_data

def load_and_filter_data():
    gene_ids, expr_matrix, expr_time_cols = load_expression_data()
    tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols = load_tf_protein_data()
    reg_map = load_regulation()

    # Filter genes: only keep those with at least one regulator.
    filtered_indices = [i for i, gene in enumerate(gene_ids) if gene in reg_map and len(reg_map[gene]) > 0]
    if len(filtered_indices) == 0:
        raise ValueError("No genes with regulators found. Exiting.")
    gene_ids = [gene_ids[i] for i in filtered_indices]
    expr_matrix = expr_matrix[filtered_indices, :]

    # For each gene, filter regulators to those present in tf_ids.
    relevant_tfs = set()
    for gene in gene_ids:
        regs = reg_map.get(gene, [])
        regs_filtered = [tf for tf in regs if tf in tf_ids]
        reg_map[gene] = regs_filtered
        relevant_tfs.update(regs_filtered)

    # Filter TFs.
    tf_ids_filtered = [tf for tf in tf_ids if tf in relevant_tfs]
    tf_ids = tf_ids_filtered
    tf_protein = {tf: tf_protein[tf] for tf in tf_ids}
    tf_psite_data = {tf: tf_psite_data[tf] for tf in tf_ids}
    tf_psite_labels = {tf: tf_psite_labels[tf] for tf in tf_ids}

    return gene_ids, expr_matrix, expr_time_cols, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols, reg_map


def prepare_data(gene_ids, expr_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols, reg_map):
    # Use the common number of time points.
    T_use = min(expr_matrix.shape[1], len(tf_time_cols))
    expr_matrix = expr_matrix[:, :T_use]
    fixed_arrays = build_fixed_arrays(gene_ids, expr_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map)
    return fixed_arrays, T_use
