import numpy as np

def build_fixed_arrays(mRNA_ids, mRNA_mat, TF_ids, protein_dict, psite_dict, psite_labels_dict, reg_map):
    """
    Builds fixed-shape arrays from the input data.
    Returns:
      - mRNA_mat: array of shape (n_mRNA, T)
      - regulators: array of shape (n_mRNA, n_reg) with indices into TF_ids.
      - protein_mat: array of shape (n_TF, T)
      - psite_tensor: array of shape (n_TF, n_psite_max, T), padded with zeros.
      - n_reg: maximum number of regulators per mRNA.
      - n_psite_max: maximum number of PSites among TFs.
      - psite_labels_arr: list (length n_TF) of lists of PSite names (padded with empty strings).
      - num_psites: array of length n_TF with the actual number of PSites for each TF.
    """
    n_mRNA, T = mRNA_mat.shape

    # Map TF_id to index.
    TF_index = {tf: idx for idx, tf in enumerate(TF_ids)}
    n_TF = len(TF_ids)

    # Determine max number of valid regulators across all mRNA, and keep valid indices only.
    reg_list = []
    for gene in mRNA_ids:
        regs = [tf for tf in reg_map.get(gene, []) if tf in TF_ids]
        reg_list.append(regs)
    n_reg = max(len(regs) for regs in reg_list) if reg_list else 1

    # Build regulators array (n_mRNA x n_reg), padded with -1 to mark invalid.
    regulators = np.full((n_mRNA, n_reg), -1, dtype=np.int32)
    for i, regs in enumerate(reg_list):
        for j, tf in enumerate(regs):
            regulators[i, j] = TF_index.get(tf, -1)

    # Build protein_mat.
    protein_mat = np.zeros((n_TF, T), dtype=np.float64)
    for tf, idx in TF_index.items():
        if protein_dict.get(tf) is not None:
            protein_mat[idx, :] = protein_dict[tf][:T]
        else:
            protein_mat[idx, :] = np.zeros(T)

    # For each TF, record the actual number of PSites.
    num_psites = np.zeros(n_TF, dtype=np.int32)
    for i, tf in enumerate(TF_ids):
        num_psites[i] = len(psite_dict.get(tf, []))
    # Maximum number of PSites across all TFs.
    n_psite_max = int(np.max(num_psites)) if np.max(num_psites) > 0 else 0

    # Build psite_tensor and psite_labels_arr.
    # psite_tensor will have shape (n_TF, n_psite_max, T) and we pad shorter vectors with zeros.
    psite_tensor = np.zeros((n_TF, n_psite_max, T), dtype=np.float64)
    psite_labels_arr = []
    for tf, idx in TF_index.items():
        psites = psite_dict.get(tf, [])
        labels = psite_labels_dict.get(tf, [])
        for j in range(n_psite_max):
            if j < len(psites):
                psite_tensor[idx, j, :] = psites[j][:T]
            else:
                psite_tensor[idx, j, :] = np.zeros(T)
        padded_labels = labels + [""] * (n_psite_max - len(labels))
        psite_labels_arr.append(padded_labels)

    return mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite_max, psite_labels_arr, num_psites
