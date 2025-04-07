import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from numba import njit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

os.makedirs('results', exist_ok=True)


# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def load_expression_data(filename="input3.csv"):
    """
    Loads gene expression (mRNA) data.
    Expects a CSV with a 'GeneID' column and time-point columns.
    """
    df = pd.read_csv(filename)
    gene_ids = df["GeneID"].astype(str).tolist()
    time_cols = [col for col in df.columns if col != "GeneID"]
    expression_matrix = df[time_cols].to_numpy(dtype=float)
    return gene_ids, expression_matrix, time_cols


def load_tf_protein_data(filename="input1_msgauss.csv"):
    """
    Loads TF protein data along with PSite information.
    Expects a CSV with 'GeneID' and 'Psite' columns.
    For rows without a valid PSite, the entire row is considered as the protein signal.
    """
    df = pd.read_csv(filename)
    tf_protein = {}
    tf_psite_data = {}
    tf_psite_labels = {}
    # Original time columns from TF data (should be 14 columns)
    orig_time_cols = [col for col in df.columns if col not in ["GeneID", "Psite"]]
    # Use only time points from index 5 onward (i.e. last 9 time points) if available.
    # To match the expression data, we need to ensure the same number of time points.
    if len(orig_time_cols) >= 14:
        time_cols = orig_time_cols[5:]
    else:
        time_cols = orig_time_cols
    for _, row in df.iterrows():
        tf = str(row["GeneID"]).strip()
        psite = str(row["Psite"]).strip()
        # Read all original values then slice.
        vals = row[orig_time_cols].to_numpy(dtype=float)
        vals = vals[5:] if len(orig_time_cols) >= 14 else vals
        if tf not in tf_protein:
            tf_protein[tf] = None
            tf_psite_data[tf] = []
            tf_psite_labels[tf] = []
        if psite == "" or psite.lower() == "nan":
            tf_protein[tf] = vals
        else:
            tf_psite_data[tf].append(vals)
            tf_psite_labels[tf].append(psite)
    tf_ids = list(tf_protein.keys())
    return tf_ids, tf_protein, tf_psite_data, tf_psite_labels, time_cols


def load_regulation(filename="input4_reduced.csv"):
    """
    Assumes the regulation file is reversed:
      - The 'Source' column holds gene (mRNA) identifiers.
      - The 'Target' column holds TF identifiers.
    Returns a mapping from gene (source) to a list of TFs (targets).
    """
    df = pd.read_csv(filename)
    reg_map = {}
    for _, row in df.iterrows():
        gene = str(row["Source"]).strip()
        tf = str(row["Target"]).strip()
        if gene not in reg_map:
            reg_map[gene] = []
        if tf not in reg_map[gene]:
            reg_map[gene].append(tf)
    return reg_map


def build_fixed_arrays(gene_ids, expression_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map):
    """
    Builds fixed-shape arrays from the input data.
    Returns:
      - expression_matrix: array of shape (n_genes, T)
      - regulators: array of shape (n_genes, n_reg) with indices into tf_ids.
      - tf_protein_matrix: array of shape (n_TF, T)
      - psite_tensor: array of shape (n_TF, n_psite_max, T), padded with zeros.
      - n_reg: maximum number of regulators per gene.
      - n_psite_max: maximum number of PSites among TFs.
      - psite_labels_arr: list (length n_TF) of lists of PSite names (padded with empty strings).
      - num_psites: array of length n_TF with the actual number of PSites for each TF.
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

    # Build tf_protein_matrix.
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


# -------------------------------
# Objective (f1)
# -------------------------------
@njit(parallel=True)
def objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
               beta_start_indices, num_psites, loss_type=0, lam1=1e-3, lam2=1e-6):
    """
    Computes a loss value using one of several loss functions.

    Parameters:
      x                  : Decision vector.
      expression_matrix  : (n_genes x T_use) measured gene expression values.
      regulators         : (n_genes x n_reg) indices of TF regulators for each gene.
      tf_protein_matrix  : (n_TF x T_use) TF protein time series.
      psite_tensor       : (n_TF x n_psite_max x T_use) matrix of PSite signals (padded with zeros).
      n_reg              : Maximum number of regulators per gene.
      T_use              : Number of time points used.
      n_genes, n_TF     : Number of genes and TF respectively.
      beta_start_indices : Integer array giving the starting index (in the β–segment) for each TF.
      num_psites         : Integer array with the actual number of PSites for each TF.
      loss_type          : Integer indicating the loss type (0: MSE, 1: MAE, 2: soft L1, 3: Cauchy, 4: Arctan, 5: Elastic Net, 6: Tikhonov).
      lam1, lam2         : Regularization parameters (used for loss_type 5 and 6).

    Returns:
      The computed loss (a scalar).
    """
    total_loss = 0.0
    n_alpha = n_genes * n_reg
    for i in prange(n_genes):
        R_meas = expression_matrix[i, :T_use]
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1:  # No valid TF for this regulator
                continue
            a = x[i * n_reg + r]
            protein = tf_protein_matrix[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]  # actual length of beta vector for TF
            beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        # For each time point, add loss according to loss_type.
        for t in range(T_use):
            e = R_meas[t] - R_pred[t]
            if loss_type == 0:  # MSE
                total_loss += e * e
            elif loss_type == 1:  # MAE
                total_loss += abs(e)
            elif loss_type == 2:  # Soft L1 (pseudo-Huber)
                total_loss += 2.0 * (np.sqrt(1.0 + e * e) - 1.0)
            elif loss_type == 3:  # Cauchy
                total_loss += np.log(1.0 + e * e)
            elif loss_type == 4:  # Arctan
                total_loss += np.arctan(e * e)
            else:
                total_loss += e * e
    loss = total_loss / (n_genes * T_use)

    # For elastic net (loss_type 5), add L1 and L2 penalties on the beta portion.
    if loss_type == 5:
        l1 = 0.0
        l2 = 0.0
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            l1 += abs(v)
            l2 += v * v
        loss += lam1 * l1 + lam2 * l2

    # For Tikhonov (loss_type 6), add L2 penalty on the beta portion.
    if loss_type == 6:
        l2 = 0.0
        for i in range(n_alpha, x.shape[0]):
            v = x[i]
            l2 += v * v
        loss += lam1 * l2

    return loss


def objective_wrapper(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                      beta_start_indices, num_psites):
    return objective_(x, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                      beta_start_indices, num_psites)


# -------------------------------
# Constraint Functions
# -------------------------------
def constraint_alpha_func(x, n_genes, n_reg):
    """
    For each gene, the sum of its α parameters must equal 1.
    """
    cons = []
    for i in range(n_genes):
        s = 0.0
        for r in range(n_reg):
            s += x[i * n_reg + r]
        cons.append(s - 1.0)
    return np.array(cons)


def constraint_beta_func(x, n_alpha, n_TF, beta_start_indices, num_psites, no_psite_tf):
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


# -------------------------------
# Plotting Functions
# -------------------------------
def compute_predictions(x, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices,
                        num_psites):
    n_alpha = n_genes * n_reg
    predictions = np.zeros((n_genes, T_use))
    for i in range(n_genes):
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1:
                continue
            a = x[i * n_reg + r]
            protein = tf_protein_matrix[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]
            beta_vec = x[n_alpha + beta_start: n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        predictions[i, :] = R_pred
    return predictions


def bootstrap_and_plot_landscape(gene_ids, expression_matrix, tf_ids, tf_protein,
                                 tf_psite_data, tf_psite_labels, reg_map, T_use,
                                 n_boot=20, contour_param_indices=(0, 5)):
    """
    Bootstraps the optimization by resampling the gene expression data (rows) with replacement,
    re-optimizes on each bootstrap sample, and then:
      (A) Generates a contour plot of the objective function landscape for two selected parameters,
          using the full-data objective function.
      (B) Plots the bootstrap distribution (via boxplots) of β-parameters grouped by TF.

    Parameters:
      gene_ids, expression_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map:
          The input data.
      T_use: Number of time points.
      n_boot: Number of bootstrap iterations.
      contour_param_indices: Tuple (i, j) giving two indices (in the decision vector) to vary for the contour plot.

    Returns:
      boot_params: A list of final parameter vectors (each from one bootstrap optimization).
      full_opt: The optimal parameter vector from the full-data optimization.
    """
    # --- Helper function to run the optimization on given (bootstrapped) data.
    def run_optimization_on_data(gene_ids_local, expression_matrix_local):
        # Filter genes: keep only those with at least one regulator.
        filtered_idx = [i for i, gene in enumerate(gene_ids_local) if gene in reg_map and len(reg_map[gene]) > 0]
        if len(filtered_idx) == 0:
            raise ValueError("No genes with regulators found in bootstrap sample.")
        gene_ids_bs = [gene_ids_local[i] for i in filtered_idx]
        expression_matrix_bs = expression_matrix_local[filtered_idx, :]

        # For each gene, filter its regulators to only those present in tf_ids.
        bs_reg_map = {}
        for gene in gene_ids_bs:
            regs = reg_map.get(gene, [])
            bs_reg_map[gene] = [tf for tf in regs if tf in tf_ids]

        # Build fixed arrays.
        (expr_mat, regulators, tf_protein_matrix, psite_tensor, n_reg,
         n_psite_max, psite_labels_arr, num_psites) = build_fixed_arrays(gene_ids_bs,
                                                                         expression_matrix_bs,
                                                                         tf_ids, tf_protein,
                                                                         tf_psite_data, tf_psite_labels,
                                                                         bs_reg_map)
        n_genes = expr_mat.shape[0]
        n_TF = tf_protein_matrix.shape[0]

        # Compute cumulative starting indices for β parameters.
        beta_start_indices = np.zeros(n_TF, dtype=np.int32)
        cum = 0
        for i in range(n_TF):
            beta_start_indices[i] = cum
            cum += 1 + num_psites[i]

        n_alpha = n_genes * n_reg
        # Build initial guess: α's uniformly and β's uniformly.
        x0_alpha = np.full(n_alpha, 1.0 / n_reg)
        x0_beta_list = []
        no_psite_tf = np.array([(num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                                for i in range(n_TF)])
        for i in range(n_TF):
            if no_psite_tf[i]:
                x0_beta_list.extend([1.0])
            else:
                length = 1 + num_psites[i]
                x0_beta_list.extend([1.0 / length] * length)
        x0_beta = np.array(x0_beta_list)
        x0 = np.concatenate([x0_alpha, x0_beta])

        # Set bounds.
        bounds_alpha = [(0.0, 1.0)] * n_alpha
        bounds_beta = [(-4.0, 4.0)] * len(x0_beta)
        bounds = bounds_alpha + bounds_beta

        # Define constraints.
        cons = [
            {"type": "eq",
             "fun": lambda x: constraint_alpha_func(x, n_genes, n_reg)},
            {"type": "eq",
             "fun": lambda x: constraint_beta_func(x, n_alpha, n_TF, beta_start_indices, num_psites, no_psite_tf)}
        ]

        # Run optimization using SLSQP.
        res = minimize(fun=objective_wrapper, x0=x0,
                       args=(
                       expr_mat, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices,
                       num_psites),
                       method="SLSQP", bounds=bounds,
                       constraints=build_linear_constraints(n_genes, n_TF, n_reg, n_alpha, beta_start_indices,
                                                            num_psites, no_psite_tf),
                       options={"disp": False, "maxiter": 20000})
        return res.x, n_genes, n_reg, beta_start_indices, num_psites, psite_labels_arr, regulators, tf_protein_matrix, psite_tensor

    # --- (1) Run the full-data optimization to obtain the reference optimum.
    full_opt, n_genes, n_reg, beta_start_indices, num_psites, psite_labels_arr, regulators, tf_protein_matrix, psite_tensor = run_optimization_on_data(
        gene_ids, expression_matrix)
    n_alpha = n_genes * n_reg

    # --- (2) Perform bootstrap re-optimizations.
    boot_params = []
    for b in range(n_boot):
        idxs = np.random.choice(len(gene_ids), size=len(gene_ids), replace=True)
        gene_ids_bs = [gene_ids[i] for i in idxs]
        expression_matrix_bs = expression_matrix[idxs, :]
        try:
            x_boot, _, _, _, _, _, _, _, _ = run_optimization_on_data(gene_ids_bs, expression_matrix_bs)
            boot_params.append(x_boot)
        except Exception as e:
            print(f"Bootstrap iteration {b} failed: {e}")

    # --- (3) Contour Plot for the Objective Landscape ---
    i1, i2 = contour_param_indices
    opt1 = full_opt[i1]
    opt2 = full_opt[i2]
    # Compute std for index i1 and i2 using only bootstrap samples that have enough length.
    vals1 = [boot[i1] for boot in boot_params if len(boot) > i1]
    vals2 = [boot[i2] for boot in boot_params if len(boot) > i2]
    std1 = np.std(vals1) if vals1 else 0
    std2 = np.std(vals2) if vals2 else 0
    grid_points = 50
    grid1 = np.linspace(opt1 - 3 * std1, opt1 + 3 * std1, grid_points)
    grid2 = np.linspace(opt2 - 3 * std2, opt2 + 3 * std2, grid_points)
    Z = np.zeros((grid_points, grid_points))
    # Evaluate objective_ (the jitted function) on a grid around the optimum.
    # Note: We use the full-data arrays (expression_matrix, regulators, etc.) from the full run.
    for idx, val1 in enumerate(grid1):
        for jdx, val2 in enumerate(grid2):
            x_temp = full_opt.copy()
            x_temp[i1] = val1
            x_temp[i2] = val2
            Z[jdx, idx] = objective_(x_temp, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg,
                                     T_use, n_genes, beta_start_indices, num_psites)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(grid1, grid2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel(f'Parameter index {i1} value')
    plt.ylabel(f'Parameter index {i2} value')
    plt.title('Objective Function Landscape (Full Data)')
    plt.show()

    # --- (4) Boxplot of β-Parameter Distributions Grouped by TF ---
    n_TF = len(beta_start_indices)
    beta_bootstrap = {}  # key: TF index, value: list of β vectors from bootstrap samples
    for tf in range(n_TF):
        start = beta_start_indices[tf]
        length = 1 + num_psites[tf]
        samples = []
        for boot in boot_params:
            if len(boot) >= n_alpha + start + length:
                samples.append(boot[n_alpha + start: n_alpha + start + length])
        if samples:
            beta_bootstrap[tf] = np.array(samples)
        else:
            beta_bootstrap[tf] = np.empty((0, length))

    box_data = []
    box_labels = []
    for tf in range(n_TF):
        if beta_bootstrap[tf].shape[0] == 0:
            continue
        for p in range(beta_bootstrap[tf].shape[1]):
            box_data.append(beta_bootstrap[tf][:, p])
            if p == 0:
                label = f'TF {tf} Protein'
            else:
                label = f'TF {tf} PSite{p}'
            box_labels.append(label)

    plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, labels=box_labels, showfliers=False)
    plt.xlabel('TF Parameters')
    plt.ylabel('Optimized Parameter Values')
    plt.title('Bootstrap Distribution of β-Parameters by TF')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return boot_params, full_opt

def plot_estimated_vs_observed(predictions, expression_matrix, gene_ids, time_points, regulators, tf_protein_matrix,
                               tf_ids,
                               num_targets=5, save_path="results"):
    """
    Plots observed and estimated gene expression time series for the first num_targets genes,
    overlaying the TF protein time series for all TFs regulating each gene.
    """
    T = len(time_points)
    time_vals_expr = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    time_vals_tf = np.array([4, 8, 16, 30, 60, 120, 240, 480, 960])
    combined_ticks = np.unique(np.concatenate((time_vals_expr, time_vals_tf)))
    num_targets = min(num_targets, predictions.shape[0])

    for i in range(num_targets):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # --- Full time series plot ---
        ax = axes[0]
        ax.plot(time_vals_expr, expression_matrix[i, :], 's-', label='Observed', color='black')
        ax.plot(time_vals_expr, predictions[i, :], '-', label='Estimated', color='red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :T]
                ax.plot(time_vals_tf, protein_signal, ':', label=f"{tf_name}")
                plotted_tfs.add(tf_name)
        ax.set_title(f"TF: {gene_ids[i]}")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fold Changes")
        ax.set_xticks(combined_ticks)
        ax.set_xticklabels(combined_ticks, rotation=45)
        ax.grid(True, alpha=0.3)

        # --- First 5 time points plot ---
        ax = axes[1]
        ax.plot(time_vals_expr[:5], expression_matrix[i, :5], 's-', label='Observed', color='black')
        ax.plot(time_vals_expr[:5], predictions[i, :5], '-', label='Estimated', color='red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :5]
                ax.plot(time_vals_tf[:5], protein_signal, ':', label=f"{tf_name}")
                plotted_tfs.add(tf_name)
        ax.set_xlabel("Time (minutes)")
        ax.set_xticks(time_vals_expr[:5])
        ax.set_xticklabels(time_vals_expr[:5], rotation=45)
        ax.legend(title="mRNAs")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # This block is for saving two plots for one TF
        # One for full time series and one for first 5 time points
        # To see clearly the dynamics early on

        # plt.figure(figsize=(8, 8))
        # plt.plot(time_vals_expr, expression_matrix[i, :], 's-', label='Observed')
        # plt.plot(time_vals_expr, predictions[i, :], '-', label='Estimated')
        # # Plot protein time series for each regulator of mRNA i (only unique TFs)
        # plotted_tfs = set()
        # for r in regulators[i, :]:
        #     if r == -1:  # Skip invalid TF
        #         continue
        #     tf_name = tf_ids[r]
        #     if tf_name not in plotted_tfs:
        #         protein_signal = tf_protein_matrix[r, :T]
        #         plt.plot(time_vals_tf, protein_signal, ':', label=f"mRNA: {tf_name}")
        #         plotted_tfs.add(tf_name)
        # plt.title(f"TF: {gene_ids[i]}")
        # plt.xlabel("Time (minutes)")
        # plt.ylabel("Fold Changes")
        # plt.xticks(combined_ticks, combined_ticks, rotation=45)
        # plt.legend()
        # plt.tight_layout()
        # plt.grid(True, alpha=0.3)
        # plt.show()
        #
        # plt.figure(figsize=(8, 8))
        # plt.plot(time_vals_expr[:5], expression_matrix[i, :5], 's-', label='Observed')
        # plt.plot(time_vals_expr[:5], predictions[i, :5], '-', label='Estimated')
        # # Plot protein time series for each regulator of mRNA i (only unique TFs)
        # plotted_tfs = set()
        # for r in regulators[i, :]:
        #     if r == -1:  # Skip invalid TF
        #         continue
        #     tf_name = tf_ids[r]
        #     if tf_name not in plotted_tfs:
        #         protein_signal = tf_protein_matrix[r, :5]
        #         plt.plot(time_vals_tf[:5], protein_signal, ':', label=f"mRNA: {tf_name}")
        #         plotted_tfs.add(tf_name)
        # plt.title(f"TF: {gene_ids[i]}")
        # plt.xlabel("Time (minutes)")
        # plt.ylabel("Fold Changes")
        # plt.xticks(time_vals_expr[:5], time_vals_expr[:5], rotation=45)
        # plt.legend()
        # plt.tight_layout()
        # plt.grid(True, alpha=0.3)
        # plt.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_vals_expr,
            y=expression_matrix[i, :],
            mode='markers+lines',
            name='Observed',
            marker=dict(symbol='square')
        ))
        fig.add_trace(go.Scatter(
            x=time_vals_expr,
            y=predictions[i, :],
            mode='lines+markers',
            name='Estimated'
        ))
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = tf_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = tf_protein_matrix[r, :len(time_vals_tf)]
                fig.add_trace(go.Scatter(
                    x=time_vals_tf,
                    y=protein_signal,
                    mode='lines',
                    name=f"mRNA: {tf_name}",
                    line=dict(dash='dot')
                ))
                plotted_tfs.add(tf_name)
        fig.update_layout(
            title=f"TF: {gene_ids[i]}",
            xaxis_title="Time (minutes)",
            yaxis_title="Fold Changes",
            xaxis=dict(
                tickmode='array',
                tickvals=combined_ticks,
                ticktext=[str(t) for t in combined_ticks]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.write_html(f"{save_path}/{gene_ids[i]}_TF_.html")


# Save results to Excel file.
def save_results_to_excel(
        gene_ids, tf_ids,
        final_alpha, final_beta, psite_labels_arr,
        expression_matrix, predictions,
        objective_value,
        reg_map,
        filename="results/results.xlsx"
):
    # --- Alpha Values ---
    alpha_rows = []
    n_genes, n_reg = final_alpha.shape
    for i in range(n_genes):
        gene = gene_ids[i]
        actual_tfs = [tf for tf in reg_map[gene] if tf in tf_ids]
        for j, tf_name in enumerate(actual_tfs):
            alpha_rows.append([gene, tf_name, final_alpha[i, j]])
    df_alpha = pd.DataFrame(alpha_rows, columns=["TF", "mRNA", "Value"])

    # --- Beta Values ---
    beta_rows = []
    for i, tf in enumerate(tf_ids):
        beta_vec = final_beta[i]
        beta_rows.append([tf, "", beta_vec[0]])  # Protein beta
        for j in range(1, len(beta_vec)):
            beta_rows.append([tf, psite_labels_arr[i][j - 1], beta_vec[j]])
    df_beta = pd.DataFrame(beta_rows, columns=["mRNA", "PSite", "Value"])

    # --- Residuals ---
    residuals = expression_matrix - predictions
    df_residuals = pd.DataFrame(residuals, columns=[f"x{j + 1}" for j in range(residuals.shape[1])])
    df_residuals.insert(0, "TF", gene_ids)

    # --- Observed ---
    df_observed = pd.DataFrame(expression_matrix, columns=[f"x{j + 1}" for j in range(expression_matrix.shape[1])])
    df_observed.insert(0, "TF", gene_ids)

    # --- Estimated ---
    df_estimated = pd.DataFrame(predictions, columns=[f"x{j + 1}" for j in range(predictions.shape[1])])
    df_estimated.insert(0, "TF", gene_ids)

    # --- Optimization Results ---
    y_true = expression_matrix.flatten()
    y_pred = predictions.flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    df_metrics = pd.DataFrame([
        ["Objective Value", objective_value],
        ["MSE", mse],
        ["MAE", mae],
        ["MAPE", mape],
        ["R^2", r2],
    ], columns=["Metric", "Value"])

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_alpha.to_excel(writer, sheet_name="Alpha Values", index=False)
        df_beta.to_excel(writer, sheet_name="Beta Values", index=False)
        df_residuals.to_excel(writer, sheet_name="Residuals", index=False)
        df_observed.to_excel(writer, sheet_name="Observed", index=False)
        df_estimated.to_excel(writer, sheet_name="Estimated", index=False)
        df_metrics.to_excel(writer, sheet_name="Optimization Results", index=False)


# -------------------------------
# Main Optimization Routine
# -------------------------------
def main():
    # Load gene expression and TF protein data.
    gene_ids, expression_matrix, expression_time_cols = load_expression_data("../data/input3.csv")
    tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols = load_tf_protein_data(
        "../data/input1_msgauss.csv")
    reg_map = load_regulation("../data/input4_reduced.csv")

    # Print time points (in minutes) for clarity.
    time_points = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    print("Time points (minutes):", time_points)

    # Filter genes: keep only those with at least one regulator.
    filtered_indices = [i for i, gene in enumerate(gene_ids) if gene in reg_map and len(reg_map[gene]) > 0]
    if len(filtered_indices) == 0:
        print("No genes with regulators found. Exiting.")
        return
    gene_ids = [gene_ids[i] for i in filtered_indices]
    expression_matrix = expression_matrix[filtered_indices, :]

    # For each gene, filter its regulators to only those present in tf_ids.
    relevant_tfs = set()
    for gene in gene_ids:
        regs = reg_map.get(gene, [])
        regs_filtered = [tf for tf in regs if tf in tf_ids]
        reg_map[gene] = regs_filtered
        relevant_tfs.update(regs_filtered)

    # Filter TFs to only those that regulate some gene.
    tf_ids_filtered = [tf for tf in tf_ids if tf in relevant_tfs]
    tf_ids = tf_ids_filtered
    tf_protein = {tf: tf_protein[tf] for tf in tf_ids}
    tf_psite_data = {tf: tf_psite_data[tf] for tf in tf_ids}
    tf_psite_labels = {tf: tf_psite_labels[tf] for tf in tf_ids}

    # Use common number of time points.
    T_use = min(expression_matrix.shape[1], len(tf_time_cols))
    expression_matrix = expression_matrix[:, :T_use]

    # Build fixed arrays.
    expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, n_psite_max, psite_labels_arr, num_psites = \
        build_fixed_arrays(gene_ids, expression_matrix, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map)
    n_genes = expression_matrix.shape[0]
    n_TF = tf_protein_matrix.shape[0]
    print(f"n_mRNAs: {n_genes}, n_TF: {n_TF}")

    # Create boolean array for TFs with no PSite data.
    no_psite_tf = np.array([(num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                            for i in range(n_TF)])

    # Compute cumulative starting indices for beta parameters for each TF.
    beta_start_indices = np.zeros(n_TF, dtype=np.int32)
    cum = 0
    for i in range(n_TF):
        beta_start_indices[i] = cum
        cum += 1 + num_psites[i]
    n_beta_total = cum

    # Build initial guess vector x0.
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
    total_dim = len(x0)

    # Set bounds.
    bounds_alpha = [(0.0, 1.0)] * n_alpha
    bounds_beta = [(-4.0, 4.0)] * len(x0_beta)
    bounds = bounds_alpha + bounds_beta

    # Define constraints.
    cons = [
        {"type": "eq",
         "fun": lambda x: constraint_alpha_func(x, n_genes, n_reg)},
        {"type": "eq",
         "fun": lambda x: constraint_beta_func(x, n_alpha, n_TF, beta_start_indices, num_psites, no_psite_tf)}
    ]

    # Run optimization using SLSQP.
    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        args=(expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices,
              num_psites),
        method="SLSQP",
        bounds=bounds,
        constraints=build_linear_constraints(n_genes, n_TF, n_reg, n_alpha, beta_start_indices, num_psites,
                                             no_psite_tf),
        options={"disp": True, "maxiter": 20000}
    )

    print("Optimization Result:")
    print(result)

    final_x = result.x
    print("\n--- Best Solution ---")
    print(f"Objective Value (F): {result.fun}")

    final_alpha = final_x[:n_alpha].reshape((n_genes, n_reg))
    final_beta = []
    for i in range(n_TF):
        start = beta_start_indices[i]
        length = 1 + num_psites[i]
        final_beta.append(final_x[n_alpha + start: n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)

    # Build mapping for α.
    alpha_mapping = {}
    for i, gene in enumerate(gene_ids):
        actual_tfs = [tf for tf in reg_map[gene] if tf in tf_ids]
        alpha_mapping[gene] = {}
        for j, tf in enumerate(actual_tfs):
            alpha_mapping[gene][tf] = final_alpha[i, j]

    print("\nMapping of TFs to mRNAs (α values):")
    for gene, mapping in alpha_mapping.items():
        print(f"{gene}:")
        for tf, a_val in mapping.items():
            print(f"   {tf}: {a_val:.4f}")

    # Build mapping for β.
    beta_mapping = {}
    for idx, tf in enumerate(tf_ids):
        beta_mapping[tf] = {}
        beta_vec = final_beta[idx]
        beta_mapping[tf][f"Protein: {tf}"] = beta_vec[0]
        for q in range(1, len(beta_vec)):
            label = psite_labels_arr[idx][q - 1]
            if label == "":
                label = f"PSite{q}"
            beta_mapping[tf][label] = beta_vec[q]
    print("\nMapping of mRNAs to β parameters:")
    for tf, mapping in beta_mapping.items():
        print(f"{tf}:")
        for label, b_val in mapping.items():
            print(f"   {label}: {b_val:.4f}")

    boot_params, full_opt = bootstrap_and_plot_landscape(gene_ids, expression_matrix, tf_ids,
                                                          tf_protein, tf_psite_data, tf_psite_labels,
                                                          reg_map, T_use, n_boot=5, contour_param_indices=(0, 5))
    # Compute predictions.
    # predictions = compute_predictions(final_x, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
    #                                   beta_start_indices, num_psites)
    # plot_estimated_vs_observed(predictions, expression_matrix, gene_ids, expression_time_cols, regulators,
    #                            tf_protein_matrix, tf_ids, num_targets=14)
    # save_results_to_excel(gene_ids, tf_ids, final_alpha, final_beta, psite_labels_arr, expression_matrix, predictions,
    #                       result.fun, reg_map)


if __name__ == "__main__":
    main()
