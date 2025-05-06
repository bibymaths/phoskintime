from tfopt.evol.utils.iodata import load_regulation, load_mRNA_data, load_TF_data


def load_raw_data():
    """
    Load raw data from files.

    Returns:
        mRNA_ids: List of mRNA gene identifiers.
        mRNA_mat: Matrix of mRNA expression data.
        mRNA_time_cols: Time points for mRNA data.
        TF_ids: List of transcription factor identifiers.
        protein_dict: Dictionary mapping TF_ids to their protein data.
        psite_dict: Dictionary mapping TF_ids to their phosphorylation site data.
        psite_labels_dict: Dictionary mapping TF_ids to their phosphorylation site labels.
        TF_time_cols: Time points for TF data.
        reg_map: Regulation map, mapping mRNA genes to their regulators.
    """
    mRNA_ids, mRNA_mat, mRNA_time_cols = load_mRNA_data()
    TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols = load_TF_data()
    reg_map = load_regulation()
    return mRNA_ids, mRNA_mat, mRNA_time_cols, TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols, reg_map


def filter_mrna(mRNA_ids, mRNA_mat, reg_map):
    """
    Filter mRNA genes to only those with regulators present in the regulation map.

    Args:
        mRNA_ids (list): List of mRNA gene identifiers.
        mRNA_mat (np.ndarray): Matrix of mRNA expression data.
        reg_map (dict): Regulation map, mapping mRNA genes to their regulators.

    Returns:
        filtered_mRNA_ids (list): List of filtered mRNA gene identifiers.
        filtered_mRNA_mat (np.ndarray): Matrix of filtered mRNA expression data.
    """
    filtered_indices = [i for i, gene in enumerate(mRNA_ids) if gene in reg_map]
    if not filtered_indices:
        raise ValueError("No mRNA with regulators found.")
    return [mRNA_ids[i] for i in filtered_indices], mRNA_mat[filtered_indices, :]


def update_regulations(mRNA_ids, reg_map, TF_ids):
    """
    Update the regulation map to only include relevant transcription factors.

    Args:
        mRNA_ids (list): List of mRNA gene identifiers.
        reg_map (dict): Regulation map, mapping mRNA genes to their regulators.
        TF_ids (list): List of transcription factor identifiers.

    Returns:
        relevant_TFs (set): Set of relevant transcription factors.
    """
    relevant_TFs = set()
    for gene in mRNA_ids:
        regs = reg_map.get(gene, [])
        reg_map[gene] = regs
        relevant_TFs.update(regs)
    return relevant_TFs


def filter_TF(TF_ids, protein_dict, psite_dict, psite_labels_dict, relevant_TFs):
    """
    Filter transcription factors to only those present in the relevant_TFs set.

    Args:
        TF_ids (list): List of transcription factor identifiers.
        protein_dict (dict): Dictionary mapping TF_ids to their protein data.
        psite_dict (dict): Dictionary mapping TF_ids to their phosphorylation site data.
        psite_labels_dict (dict): Dictionary mapping TF_ids to their phosphorylation site labels.
        relevant_TFs (set): Set of relevant transcription factors.

    Returns:
        TF_ids_filtered (list): List of filtered transcription factor identifiers.
        protein_dict (dict): Filtered dictionary mapping TF_ids to their protein data.
        psite_dict (dict): Filtered dictionary mapping TF_ids to their phosphorylation site data.
        psite_labels_dict (dict): Filtered dictionary mapping TF_ids to their phosphorylation site labels.
    """
    TF_ids_filtered = [tf for tf in TF_ids if tf in relevant_TFs]
    protein_dict = {tf: protein_dict[tf] for tf in TF_ids_filtered}
    psite_dict = {tf: psite_dict[tf] for tf in TF_ids_filtered}
    psite_labels_dict = {tf: psite_labels_dict[tf] for tf in TF_ids_filtered}
    return TF_ids_filtered, protein_dict, psite_dict, psite_labels_dict


def determine_T_use(mRNA_mat, TF_time_cols):
    """
    Determine the number of time points to use for the analysis.

    Args:
        mRNA_mat (np.ndarray): Matrix of mRNA expression data.
        TF_time_cols (list): Time points for TF data.
    """
    T_use = min(mRNA_mat.shape[1], len(TF_time_cols))
    return T_use
