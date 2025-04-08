
from tfopt.evol.utils.iodata import load_regulation, load_mRNA_data, load_TF_data

# -------------------------------
# Data Preprocessing and Loading
# -------------------------------
def load_raw_data():
    mRNA_ids, mRNA_mat, mRNA_time_cols = load_mRNA_data()
    TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols = load_TF_data()
    reg_map = load_regulation()
    return mRNA_ids, mRNA_mat, mRNA_time_cols, TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols, reg_map

def filter_mrna(mRNA_ids, mRNA_mat, reg_map):
    filtered_indices = [i for i, gene in enumerate(mRNA_ids) if gene in reg_map and len(reg_map[gene]) > 0]
    if not filtered_indices:
        raise ValueError("No mRNA with regulators found.")
    return [mRNA_ids[i] for i in filtered_indices], mRNA_mat[filtered_indices, :]

def update_regulations(mRNA_ids, reg_map, TF_ids):
    """For each mRNA, filter its regulators to only those present in TF_ids and build a set of all relevant TFs."""
    relevant_TFs = set()
    for gene in mRNA_ids:
        regs = reg_map.get(gene, [])
        regs_filtered = [tf for tf in regs if tf in TF_ids]
        reg_map[gene] = regs_filtered
        relevant_TFs.update(regs_filtered)
    return relevant_TFs

def filter_TF(TF_ids, protein_dict, psite_dict, psite_labels_dict, relevant_TFs):
    TF_ids_filtered = [tf for tf in TF_ids if tf in relevant_TFs]
    protein_dict = {tf: protein_dict[tf] for tf in TF_ids_filtered}
    psite_dict = {tf: psite_dict[tf] for tf in TF_ids_filtered}
    psite_labels_dict = {tf: psite_labels_dict[tf] for tf in TF_ids_filtered}
    return TF_ids_filtered, protein_dict, psite_dict, psite_labels_dict

def determine_T_use(mRNA_mat, TF_time_cols):
    T_use = min(mRNA_mat.shape[1], len(TF_time_cols))
    return T_use