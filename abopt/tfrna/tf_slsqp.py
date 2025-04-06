#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from numba import njit, prange
import matplotlib.pyplot as plt

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def load_mRNA_data(filename="input3.csv"):
    """
    Loads mRNA time series from input3.csv.
    Assumes first column is 'GeneID' and remaining columns are time points.
    Returns:
      - mRNA_ids: list of gene IDs (strings)
      - mRNA_mat: NumPy array of shape (n_mRNA, T) with measured values.
      - time_cols: list of time point column names.
    """
    df = pd.read_csv(filename)
    mRNA_ids = df["GeneID"].astype(str).tolist()
    time_cols = [col for col in df.columns if col != "GeneID"]
    mRNA_mat = df[time_cols].to_numpy(dtype=float)
    return mRNA_ids, mRNA_mat, time_cols

def load_TF_data(filename="input1_msgauss.csv"):
    """
    Loads TF data from input1_msgauss.csv.
    Assumes columns: 'GeneID', 'Psite', and time series columns (e.g., x1, x2, ...).
    For each TF, rows with empty 'Psite' give the protein time series;
    rows with nonempty 'Psite' give PSite time series.
    In addition, the PSite label (its string value) is stored.
    Returns:
      - TF_ids: list of unique TF IDs.
      - protein_dict: dict mapping TF_id -> protein array (1D, length T)
      - psite_dict: dict mapping TF_id -> list of PSite arrays (each 1D, length T)
      - psite_labels_dict: dict mapping TF_id -> list of PSite names.
      - time_cols: list of time point column names.
    """
    df = pd.read_csv(filename)
    protein_dict = {}
    psite_dict = {}
    psite_labels_dict = {}
    for _, row in df.iterrows():
        tf = str(row["GeneID"]).strip()
        psite = str(row["Psite"]).strip()
        time_cols = [col for col in df.columns if col not in ["GeneID", "Psite"]]
        vals = row[time_cols].to_numpy(dtype=float)
        if tf not in protein_dict:
            protein_dict[tf] = None
            psite_dict[tf] = []
            psite_labels_dict[tf] = []
        if psite == "" or psite.lower() == "nan":
            protein_dict[tf] = vals
        else:
            psite_dict[tf].append(vals)
            psite_labels_dict[tf].append(psite)
    TF_ids = list(protein_dict.keys())
    return TF_ids, protein_dict, psite_dict, psite_labels_dict, time_cols

def load_regulation(filename="input4_reduced.csv"):
    """
    Loads TF–mRNA interactions from input4_reduced.csv.
    Assumes columns: 'Source' and 'Target'.
    Returns:
      - reg_map: dict mapping target (mRNA) -> list of regulating TF IDs.
    """
    df = pd.read_csv(filename)
    reg_map = {}
    for _, row in df.iterrows():
        source = str(row["Source"]).strip()
        target = str(row["Target"]).strip()
        if target not in reg_map:
            reg_map[target] = []
        if source not in reg_map[target]:
            reg_map[target].append(source)
    return reg_map

def build_fixed_arrays(mRNA_ids, mRNA_mat, TF_ids, protein_dict, psite_dict, psite_labels_dict, reg_map):
    """
    Builds fixed‐shape arrays from the input data.
    For each mRNA, we use its regulators from reg_map.
      - n_reg: maximum number of regulators among all mRNA targets.
    For each TF, let n_psite be the maximum number of PSites among TFs that have PSite data.
    For TFs that lack any PSite, we will still allocate n_psite columns but later add extra
    constraints to force the PSite β's to zero.
    Returns:
      - mRNA_mat: array (n_mRNA, T)
      - regulators: array (n_mRNA, n_reg) with indices into TF_ids.
      - protein_mat: array (n_TF, T) for TF protein time series.
      - psite_tensor: array (n_TF, n_psite, T) for PSite data (padded with zeros).
      - n_reg, n_psite, and
      - psite_labels_arr: list (length n_TF) of lists of PSite names (padded with empty strings).
    """
    n_mRNA, T = mRNA_mat.shape

    # Build mapping from TF_id to index
    TF_index = {tf: idx for idx, tf in enumerate(TF_ids)}
    n_TF = len(TF_ids)

    # Determine maximum number of regulators per mRNA.
    max_reg = 0
    reg_list = []
    for gene in mRNA_ids:
        regs = reg_map.get(gene, [])
        max_reg = max(max_reg, len(regs))
        reg_list.append(regs)
    n_reg = max_reg if max_reg > 0 else 1

    # Build regulators array (n_mRNA x n_reg), padding with index 0.
    regulators = np.zeros((n_mRNA, n_reg), dtype=np.int32)
    for i, regs in enumerate(reg_list):
        for j in range(n_reg):
            if j < len(regs):
                regulators[i, j] = TF_index.get(regs[j], 0)
            else:
                regulators[i, j] = 0

    # Build protein_mat: for each TF, get its protein measurement.
    protein_mat = np.zeros((n_TF, T), dtype=np.float64)
    for tf, idx in TF_index.items():
        if protein_dict.get(tf) is not None:
            protein_mat[idx, :] = protein_dict[tf][:T]
        else:
            protein_mat[idx, :] = np.zeros(T)

    # Determine maximum number of PSites among TFs (if any have PSite data).
    max_psite = 0
    for tf in TF_ids:
        n = len(psite_dict.get(tf, []))
        max_psite = max(max_psite, n)
    n_psite = max_psite  # Note: if none have PSite, n_psite==0.

    # Build psite_tensor and psite_labels_arr.
    # For each TF, if no PSite data is present, allocate an array of shape (n_psite, T) of zeros
    # and a list of n_psite empty strings.
    psite_tensor = np.zeros((n_TF, n_psite, T), dtype=np.float64)
    psite_labels_arr = []
    for tf, idx in TF_index.items():
        psites = psite_dict.get(tf, [])
        labels = psite_labels_dict.get(tf, [])
        for j in range(n_psite):
            if j < len(psites):
                psite_tensor[idx, j, :] = psites[j][:T]
            else:
                psite_tensor[idx, j, :] = np.zeros(T)
        padded_labels = labels + [""] * (n_psite - len(labels))
        psite_labels_arr.append(padded_labels)

    return mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, psite_labels_arr

# -------------------------------
# Numba-Accelerated Objective
# -------------------------------
@njit(parallel=True)
def objective_numba_fixed(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA, n_TF):
    """
    Computes the sum of squared errors.
    x is a flat vector of parameters:
      - First n_mRNA*n_reg entries are α parameters.
      - Next n_TF*(1+n_psite) entries are β parameters for each TF.
    """
    total_error = 0.0
    n_alpha = n_mRNA * n_reg
    for i in prange(n_mRNA):
        R_meas = mRNA_mat[i, :T_use]
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
            beta_start = n_alpha + tf_idx * (1 + n_psite)
            beta_vec = x[beta_start : beta_start + 1 + n_psite]
            tf_effect = beta_vec[0] * protein
            for k in range(n_psite):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        for t in range(T_use):
            diff = R_meas[t] - R_pred[t]
            total_error += diff * diff
    return total_error

def objective_wrapper(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA, n_TF):
    return objective_numba_fixed(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA, n_TF)

# -------------------------------
# Constraint Functions
# -------------------------------
def constraint_alpha_func(x, n_mRNA, n_reg):
    """
    For each mRNA, the sum of its α parameters must equal 1.
    """
    cons = []
    for i in range(n_mRNA):
        s = 0.0
        for r in range(n_reg):
            s += x[i * n_reg + r]
        cons.append(s - 1.0)
    return np.array(cons)

def constraint_beta_func_extended(x, n_alpha, n_TF, n_psite, no_psite_tf):
    """
    For each TF, the sum of its β parameters must equal 1.
    Additionally, for TFs with no PSite data (as indicated by no_psite_tf),
    the additional β parameters (for PSites) must be 0.
    """
    cons = []
    for tf in range(n_TF):
        start = n_alpha + tf * (1 + n_psite)
        beta_vec = x[start : start + (1 + n_psite)]
        cons.append(np.sum(beta_vec) - 1.0)
        if no_psite_tf[tf]:
            # Force all PSite β's to 0.
            for q in range(1, 1 + n_psite):
                cons.append(beta_vec[q])
    return np.array(cons)

# -------------------------------
# Plotting Functions
# -------------------------------
def compute_predictions(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA):
    """
    Computes predicted mRNA time series for each mRNA.
    """
    n_alpha = n_mRNA * n_reg
    predictions = np.zeros((n_mRNA, T_use))
    for i in range(n_mRNA):
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
            beta_start = n_alpha + tf_idx * (1 + n_psite)
            beta_vec = x[beta_start : beta_start + 1 + n_psite]
            tf_effect = beta_vec[0] * protein
            for k in range(n_psite):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        predictions[i, :] = R_pred
    return predictions


def plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, time_points, regulators, protein_mat, TF_ids,
                               num_targets=5):
    """
    Plots observed and estimated mRNA time series for the first num_targets mRNAs,
    and overlays the protein time series for all TFs regulating each mRNA.

    Parameters:
      predictions: (n_mRNA x T) array of estimated mRNA values.
      mRNA_mat: (n_mRNA x T) array of observed mRNA values.
      mRNA_ids: list of mRNA target identifiers.
      time_points: list (or array) of time point labels.
      regulators: (n_mRNA x n_reg) array; each row contains indices (into TF_ids) of regulators for that mRNA.
      protein_mat: (n_TF x T) array containing TF protein time series.
      TF_ids: list of TF identifiers corresponding to protein_mat rows.
      num_targets: number of mRNA targets to plot.
    """
    T = len(time_points)
    time_vals = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960]) #np.arange(T)
    num_targets = min(num_targets, predictions.shape[0])
    plt.figure(figsize=(10, num_targets * 3))
    for i in range(num_targets):
        plt.subplot(num_targets, 1, i + 1)
        plt.plot(time_vals, mRNA_mat[i, :], 'o-', label='Observed')
        plt.plot(time_vals, predictions[i, :], 's--', label='Estimated')
        # Plot protein time series for each regulator of mRNA i (only unique TFs)
        plotted_tfs = set()
        for r in regulators[i, :]:
            tf_name = TF_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = protein_mat[r, :T]
                plt.plot(time_vals, protein_signal, ':', label=f"TF: {tf_name}")
                plotted_tfs.add(tf_name)
        plt.title(f"mRNA: {mRNA_ids[i]}")
        plt.xlabel("Time Point")
        plt.ylabel("Expression")
        plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main Optimization Routine
# -------------------------------
def main():
    # Load raw data
    mRNA_ids, mRNA_mat, mRNA_time_cols = load_mRNA_data("../data/input3.csv")
    TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols = load_TF_data("../data/input1_msgauss.csv")
    reg_map = load_regulation("../data/input4_reduced.csv")

    # Filter mRNA: keep only those with at least one regulator.
    filtered_indices = [i for i, gene in enumerate(mRNA_ids) if gene in reg_map and len(reg_map[gene]) > 0]
    if len(filtered_indices) == 0:
        print("No mRNA with regulators found. Exiting.")
        return
    mRNA_ids = [mRNA_ids[i] for i in filtered_indices]
    mRNA_mat = mRNA_mat[filtered_indices, :]

    # For each mRNA, filter its regulators to only those present in TF_ids.
    relevant_TFs = set()
    for gene in mRNA_ids:
        regs = reg_map.get(gene, [])
        regs_filtered = [tf for tf in regs if tf in TF_ids]
        reg_map[gene] = regs_filtered
        relevant_TFs.update(regs_filtered)

    # Filter TFs to only those that actually regulate some mRNA.
    TF_ids_filtered = [tf for tf in TF_ids if tf in relevant_TFs]
    TF_ids = TF_ids_filtered
    protein_dict = {tf: protein_dict[tf] for tf in TF_ids}
    psite_dict = {tf: psite_dict[tf] for tf in TF_ids}
    psite_labels_dict = {tf: psite_labels_dict[tf] for tf in TF_ids}

    # Use common number of time points (T_use)
    T_use = min(mRNA_mat.shape[1], len(TF_time_cols))
    mRNA_mat = mRNA_mat[:, :T_use]

    # Build fixed arrays.
    mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, psite_labels_arr = \
        build_fixed_arrays(mRNA_ids, mRNA_mat, TF_ids, protein_dict, psite_dict, psite_labels_dict, reg_map)
    n_mRNA = mRNA_mat.shape[0]
    n_TF = protein_mat.shape[0]
    print(f"n_mRNA: {n_mRNA}, n_TF: {n_TF}, T_use: {T_use}, n_reg: {n_reg}, n_psite: {n_psite}")

    # Compute a boolean array indicating which TFs have no PSite data.
    no_psite_tf = np.array([all(label == "" for label in labels) for labels in psite_labels_arr])

    # Build initial guess vector x0:
    n_alpha = n_mRNA * n_reg
    x0_alpha = np.full(n_alpha, 1.0 / n_reg)
    n_beta = n_TF * (1 + n_psite)
    x0_beta = np.full(n_beta, 1.0 / (1 + n_psite))
    x0 = np.concatenate([x0_alpha, x0_beta])
    total_dim = len(x0)
    print(f"Total parameter dimension: {total_dim}")

    # Set bounds.
    bounds_alpha = [(0.0, 1.0)] * n_alpha
    bounds_beta = [(-4.0, 4.0)] * n_beta
    bounds = bounds_alpha + bounds_beta

    # Define constraints: both the sum constraints and for TFs with no PSite data, force extra β's to zero.
    cons = [
        {"type": "eq",
         "fun": lambda x: constraint_alpha_func(x, n_mRNA, n_reg)},
        {"type": "eq",
         "fun": lambda x: constraint_beta_func_extended(x, n_alpha, n_TF, n_psite, no_psite_tf)}
    ]

    # Run optimization using SLSQP.
    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        args=(mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA, n_TF),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"disp": True, "maxiter": 20000}
    )

    print("Optimization Result:")
    print(result)

    # --- Display Results with Detailed Mapping ---
    final_x = result.x
    final_alpha = final_x[:n_alpha].reshape((n_mRNA, n_reg))
    final_beta = final_x[n_alpha:].reshape((n_TF, 1 + n_psite))

    # Build mapping for α: each mRNA mapped to its regulating TFs and corresponding α.
    alpha_mapping = {}
    for i, mRNA in enumerate(mRNA_ids):
        alpha_mapping[mRNA] = {}
        for j in range(n_reg):
            tf_idx = regulators[i, j]
            tf_name = TF_ids[tf_idx]
            alpha_mapping[mRNA][tf_name] = final_alpha[i, j]
    print("\nMapping of mRNA targets to regulators (α values):")
    for mRNA, mapping in alpha_mapping.items():
        print(f"{mRNA}:")
        for tf, a_val in mapping.items():
            print(f"   {tf}: {a_val:.4f}")

    # Build mapping for β:
    # For each TF, label β0 as 'Protein: <TF name>'.
    # For PSite entries, use the corresponding PSite name from psite_labels_arr.
    beta_mapping = {}
    for idx, tf in enumerate(TF_ids):
        beta_mapping[tf] = {}
        beta_mapping[tf][f"Protein: {tf}"] = final_beta[idx, 0]
        for q in range(1, 1 + n_psite):
            label = psite_labels_arr[idx][q - 1]
            if label == "":
                label = f"PSite{q}"
            beta_mapping[tf][label] = final_beta[idx, q]
    print("\nMapping of TFs to β parameters (interpreted as relative impacts):")
    for tf, mapping in beta_mapping.items():
        print(f"{tf}:")
        for label, b_val in mapping.items():
            print(f"   {label}: {b_val:.4f}")

    # Compute predictions using the final parameter vector final_x.
    predictions = compute_predictions(final_x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA)
    # Plot observed vs estimated mRNA time series with overlaid TF protein signals.
    plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, mRNA_time_cols, regulators, protein_mat, TF_ids, num_targets=15)

if __name__ == "__main__":
    main()