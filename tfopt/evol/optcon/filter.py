
from tfopt.evol.utils.iodata import load_regulation, load_mRNA_data, load_TF_data


def load_raw_data():
    """
    Load raw data from files.
    This includes mRNA data, TF data, and regulation maps.
    The function returns the following:
    - mRNA_ids: List of mRNA gene identifiers.
    - mRNA_mat: Matrix of mRNA expression data.
    - mRNA_time_cols: Time points for mRNA data.
    - TF_ids: List of transcription factor identifiers.
    - protein_dict: Dictionary mapping TF_ids to their protein data.
    - psite_dict: Dictionary mapping TF_ids to their phosphorylation site data.
    - psite_labels_dict: Dictionary mapping TF_ids to their phosphorylation site labels.
    - TF_time_cols: Time points for TF data.
    - reg_map: Regulation map, mapping mRNA genes to their regulators.

    :return: mRNA_ids, mRNA_mat, mRNA_time_cols, TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols, reg_map
    """
    mRNA_ids, mRNA_mat, mRNA_time_cols = load_mRNA_data()
    TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols = load_TF_data()
    reg_map = load_regulation()
    return mRNA_ids, mRNA_mat, mRNA_time_cols, TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols, reg_map

def filter_mrna(mRNA_ids, mRNA_mat, reg_map):
    """
    Filter mRNA genes to only those with regulators present in the regulation map.
    This function returns the filtered mRNA_ids and their corresponding expression matrix.

    :param mRNA_ids: List of mRNA gene identifiers.
    :param mRNA_mat: Matrix of mRNA expression data.
    :param reg_map: Regulation map, mapping mRNA genes to their regulators.
    :return: filtered_mRNA_ids, filtered_mRNA_mat
    """
    filtered_indices = [i for i, gene in enumerate(mRNA_ids) if gene in reg_map]
    if not filtered_indices:
        raise ValueError("No mRNA with regulators found.")
    return [mRNA_ids[i] for i in filtered_indices], mRNA_mat[filtered_indices, :]

def update_regulations(mRNA_ids, reg_map, TF_ids):
    """
    Update the regulation map to only include relevant transcription factors.
    This function modifies the reg_map in place and returns a set of relevant transcription factors.

    :param mRNA_ids: List of mRNA gene identifiers.
    :param reg_map: Regulation map, mapping mRNA genes to their regulators.
    :param TF_ids: List of transcription factor identifiers.
    :return: relevant_TFs: Set of relevant transcription factors.
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
    This function returns the filtered TF_ids and their corresponding protein and phosphorylation site data.
    This is important for ensuring that only relevant transcription factors are included in the analysis.
    :param TF_ids: List of transcription factor identifiers.
    :param protein_dict: Dictionary mapping TF_ids to their protein data.
    :param psite_dict: Dictionary mapping TF_ids to their phosphorylation site data.
    :param psite_labels_dict: Dictionary mapping TF_ids to their phosphorylation site labels.
    :param relevant_TFs: Set of relevant transcription factors.
    :return: filtered TF_ids, protein_dict, psite_dict, psite_labels_dict
    """
    TF_ids_filtered = [tf for tf in TF_ids if tf in relevant_TFs]
    protein_dict = {tf: protein_dict[tf] for tf in TF_ids_filtered}
    psite_dict = {tf: psite_dict[tf] for tf in TF_ids_filtered}
    psite_labels_dict = {tf: psite_labels_dict[tf] for tf in TF_ids_filtered}
    return TF_ids_filtered, protein_dict, psite_dict, psite_labels_dict

def determine_T_use(mRNA_mat, TF_time_cols):
    """
    Determine the number of time points to use for the analysis.
    This function takes the mRNA matrix and TF time columns as input
    and returns the minimum number of time points available across both datasets.
    This is important for ensuring that the analysis is consistent and
    comparable across different datasets.
    The function checks the shape of the mRNA matrix and the length of the TF time columns
    to determine the minimum number of time points.
    If the mRNA matrix has fewer time points than the TF time columns,
    it uses the number of time points in the mRNA matrix.
    If the TF time columns have fewer time points, it uses that value instead.
    This ensures that the analysis is based on the same number of time points
    for both mRNA and TF data.

    :param mRNA_mat: Matrix of mRNA expression data.
    :param TF_time_cols: Time points for TF data.
    :return: T_use: Number of time points to use for the analysis.
    """
    T_use = min(mRNA_mat.shape[1], len(TF_time_cols))
    return T_use