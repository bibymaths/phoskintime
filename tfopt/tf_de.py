import subprocess
from multiprocessing.pool import ThreadPool

import pandas as pd
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem, StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def load_mRNA_data(filename="input3.csv"):
    df = pd.read_csv(filename)
    mRNA_ids = df["GeneID"].astype(str).tolist()
    time_cols = [col for col in df.columns if col != "GeneID"]
    mRNA_mat = df[time_cols].to_numpy(dtype=float)
    return mRNA_ids, mRNA_mat, time_cols

def load_TF_data(filename="input1_msgauss.csv"):
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
    Assumes the regulation file is reversed:
      - The 'Source' column holds mRNA identifiers.
      - The 'Target' column holds TF identifiers.
    Returns a mapping from mRNA (source) to a list of TFs (targets).
    """
    df = pd.read_csv(filename)
    reg_map = {}
    for _, row in df.iterrows():
        mrna = str(row["Source"]).strip()
        tf = str(row["Target"]).strip()
        if mrna not in reg_map:
            reg_map[mrna] = []
        if tf not in reg_map[mrna]:
            reg_map[mrna].append(tf)
    return reg_map

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

    # # Determine maximum number of regulators per mRNA.
    # max_reg = 0
    # reg_list = []
    # for gene in mRNA_ids:
    #     regs = reg_map.get(gene, [])
    #     max_reg = max(max_reg, len(regs))
    #     reg_list.append(regs)
    # n_reg = max_reg if max_reg > 0 else 1
    #
    # # Build regulators array (n_mRNA x n_reg), padded with index 0.
    # regulators = np.zeros((n_mRNA, n_reg), dtype=np.int32)
    # for i, regs in enumerate(reg_list):
    #     for j in range(n_reg):
    #         if j < len(regs):
    #             regulators[i, j] = TF_index.get(regs[j], 0)
    #         else:
    #             regulators[i, j] = 0

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

# -------------------------------
# Objective (f1)
# -------------------------------
"""
@njit(parallel=True)
def objective_(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA, beta_start_indices, num_psites):
    total_error = 0.0
    n_alpha = n_mRNA * n_reg
    for i in prange(n_mRNA):
        R_meas = mRNA_mat[i, :T_use]
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]
            beta_vec = x[n_alpha + beta_start : n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        for t in range(T_use):
            diff = R_meas[t] - R_pred[t]
            total_error += diff * diff
    return total_error / (T_use * n_mRNA)
"""
# Loss functions for objective function.
# 0: MSE, 1: MAE, 2: soft L1 (pseudo-Huber), 3: Cauchy, 4: Arctan, 5: Elastic Net, 6: Tikhonov.
# The loss functions are implemented in the objective_ function.
# The loss function is selected using the loss_type parameter.
# The default is MSE (0).
@njit(parallel=True)
def objective_(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA,
                   beta_start_indices, num_psites, loss_type=0, lam1=1e-3, lam2=1e-3):
    """
    Computes a loss value using one of several loss functions.

    Parameters:
      x               : Decision vector.
      mRNA_mat        : (n_mRNA x T_use) measured mRNA values.
      regulators      : (n_mRNA x n_reg) indices of TF regulators for each mRNA.
      protein_mat     : (n_TF x T_use) TF protein time series.
      psite_tensor    : (n_TF x n_psite_max x T_use) matrix of PSite signals (padded with zeros).
      n_reg           : Maximum number of regulators per mRNA.
      T_use           : Number of time points used.
      n_mRNA, n_TF    : Number of mRNA and TF respectively.
      beta_start_indices: Integer array giving the starting index (in the β–segment)
                         for each TF.
      num_psites      : Integer array with the actual number of PSites for each TF.
      loss_type       : Integer indicating the loss type (0: MSE, 1: MAE, 2: soft L1,
                         3: Cauchy, 4: Arctan, 5: Elastic Net, 6: Tikhonov).
      lam1, lam2      : Regularization parameters (used for loss_type 5 and 6).

    Returns:
      The computed loss (a scalar).
    """
    total_loss = 0.0
    n_alpha = n_mRNA * n_reg
    for i in prange(n_mRNA):
        R_meas = mRNA_mat[i, :T_use]
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1: # No valid TF for this regulator
                continue
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
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
                # Default to MSE if unknown.
                total_loss += e * e
    loss = total_loss / (n_mRNA * T_use)

    # For elastic net (loss_type 5), add L1 and L2 penalties on the beta portion.
    if loss_type == 5:
        l1 = 0.0
        l2 = 0.0
        # Compute over beta parameters only.
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

# -------------------------------
# Plotting Functions
# -------------------------------
def compute_predictions(x, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA, beta_start_indices, num_psites):
    n_alpha = n_mRNA * n_reg
    predictions = np.zeros((n_mRNA, T_use))
    for i in range(n_mRNA):
        R_pred = np.zeros(T_use)
        for r in range(n_reg):
            tf_idx = regulators[i, r]
            if tf_idx == -1: # No valid TF for this regulator
                continue
            a = x[i * n_reg + r]
            protein = protein_mat[tf_idx, :T_use]
            beta_start = beta_start_indices[tf_idx]
            length = 1 + num_psites[tf_idx]
            beta_vec = x[n_alpha + beta_start : n_alpha + beta_start + length]
            tf_effect = beta_vec[0] * protein
            for k in range(num_psites[tf_idx]):
                tf_effect += beta_vec[k + 1] * psite_tensor[tf_idx, k, :T_use]
            R_pred += a * tf_effect
        predictions[i, :] = R_pred
    return predictions

def plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, time_points, regulators, protein_mat, TF_ids,
                               num_targets=5, save_path="results"):
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
    time_vals_mrna = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    time_vals_tf = np.array([4, 8, 16, 30, 60, 120, 240, 480, 960])
    combined_ticks = np.unique(np.concatenate((time_vals_mrna, time_vals_tf)))
    num_targets = min(num_targets, predictions.shape[0])

    for i in range(num_targets):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # two side-by-side plots

        # --- Full time series plot ---
        ax = axes[0]
        ax.plot(time_vals_mrna, mRNA_mat[i, :], 's-', label='Observed', color = 'black')
        ax.plot(time_vals_mrna, predictions[i, :], '-', label='Estimated', color = 'red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = TF_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = protein_mat[r, :T]
                ax.plot(time_vals_tf, protein_signal, ':', label=f"{tf_name}")
                plotted_tfs.add(tf_name)
        ax.set_title(f"{mRNA_ids[i]}")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fold Changes")
        ax.set_xticks(combined_ticks)
        ax.set_xticklabels(combined_ticks, rotation=45)
        # ax.legend()
        ax.grid(True, alpha=0.3)

        # --- First 5 time points plot ---
        ax = axes[1]
        ax.plot(time_vals_mrna[:5], mRNA_mat[i, :5], 's-', label='Observed', color = 'black')
        ax.plot(time_vals_mrna[:5], predictions[i, :5], '-', label='Estimated', color = 'red')
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:
                continue
            tf_name = TF_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = protein_mat[r, :5]
                ax.plot(time_vals_tf[:5], protein_signal, ':', label=f"{tf_name}")
                plotted_tfs.add(tf_name)
        # ax.set_title(f"mRNA: {mRNA_ids[i]}")
        ax.set_xlabel("Time (minutes)")
        # ax.set_ylabel("Fold Changes")
        ax.set_xticks(time_vals_mrna[:5])
        ax.set_xticklabels(time_vals_mrna[:5], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # for i in range(num_targets):
    #     plt.figure(figsize=(8, 8))
    #     plt.plot(time_vals_mrna, mRNA_mat[i, :], 's-', label='Observed')
    #     plt.plot(time_vals_mrna, predictions[i, :], '-', label='Estimated')
    #     # Plot protein time series for each regulator of mRNA i (only unique TFs)
    #     plotted_tfs = set()
    #     for r in regulators[i, :]:
    #         if r == -1:  # Skip invalid TF
    #             continue
    #         tf_name = TF_ids[r]
    #         if tf_name not in plotted_tfs:
    #             protein_signal = protein_mat[r, :T]
    #             plt.plot(time_vals_tf, protein_signal, ':', label=f"TF: {tf_name}")
    #             plotted_tfs.add(tf_name)
    #     plt.title(f"mRNA: {mRNA_ids[i]}")
    #     plt.xlabel("Time (minutes)")
    #     plt.ylabel("Fold Changes")
    #     plt.xticks(combined_ticks, combined_ticks, rotation=45)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.grid(True, alpha=0.3)
    #     plt.show()
    #
    #     plt.figure(figsize=(8, 8))
    #     plt.plot(time_vals_mrna[:5], mRNA_mat[i, :5], 's-', label='Observed')
    #     plt.plot(time_vals_mrna[:5], predictions[i, :5], '-', label='Estimated')
    #     # Plot protein time series for each regulator of mRNA i (only unique TFs)
    #     plotted_tfs = set()
    #     for r in regulators[i, :]:
    #         if r == -1:  # Skip invalid TF
    #             continue
    #         tf_name = TF_ids[r]
    #         if tf_name not in plotted_tfs:
    #             protein_signal = protein_mat[r, :5]
    #             plt.plot(time_vals_tf[:5], protein_signal, ':', label=f"TF: {tf_name}")
    #             plotted_tfs.add(tf_name)
    #     plt.title(f"mRNA: {mRNA_ids[i]}")
    #     plt.xlabel("Time (minutes)")
    #     plt.ylabel("Fold Changes")
    #     plt.xticks(time_vals_mrna[:5], time_vals_mrna[:5], rotation=45)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.grid(True, alpha=0.3)
    #     plt.show()

        fig = go.Figure()
        # Observed mRNA
        fig.add_trace(go.Scatter(
            x=time_vals_mrna,
            y=mRNA_mat[i, :],
            mode='markers+lines',
            name='Observed',
            marker=dict(symbol='square')
        ))
        # Estimated mRNA
        fig.add_trace(go.Scatter(
            x=time_vals_mrna,
            y=predictions[i, :],
            mode='lines+markers',
            name='Estimated'
        ))
        # TF protein signals: plot each unique regulator only once.
        plotted_tfs = set()
        for r in regulators[i, :]:
            if r == -1:  # Skip invalid TF
                continue
            tf_name = TF_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = protein_mat[r, :len(time_vals_tf)]
                fig.add_trace(go.Scatter(
                    x=time_vals_tf,
                    y=protein_signal,
                    mode='lines',
                    name=f"TF: {tf_name}",
                    line=dict(dash='dot')
                ))
                plotted_tfs.add(tf_name)
        fig.update_layout(
            title=f"mRNA: {mRNA_ids[i]}",
            xaxis_title="Time (minutes)",
            yaxis_title="Fold Changes",
            xaxis=dict(
                tickmode='array',
                tickvals=combined_ticks,
                ticktext=[str(t) for t in combined_ticks]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.write_html(f"{save_path}/mRNA_{mRNA_ids[i]}.html")

# Save results to Excel file.
def save_results_to_excel(
    mRNA_ids, TF_ids,
    final_alpha, final_beta, psite_labels_arr,
    mRNA_mat, predictions,
    objective_value,
    reg_map,
    filename="results/results.xlsx"
):
    # --- Alpha Values ---
    alpha_rows = []
    n_mRNA, n_reg = final_alpha.shape
    for i in range(n_mRNA):
        gene = mRNA_ids[i]
        actual_tfs = [tf for tf in reg_map[gene] if tf in TF_ids]
        for j, tf_name in enumerate(actual_tfs):
            alpha_rows.append([gene, tf_name, final_alpha[i, j]])
    df_alpha = pd.DataFrame(alpha_rows, columns=["TF", "mRNA", "Value"])

    # --- Beta Values ---
    beta_rows = []
    for i, tf in enumerate(TF_ids):
        beta_vec = final_beta[i]
        beta_rows.append([tf, "", beta_vec[0]])  # Protein beta
        for j in range(1, len(beta_vec)):
            beta_rows.append([tf, psite_labels_arr[i][j - 1], beta_vec[j]])
    df_beta = pd.DataFrame(beta_rows, columns=["mRNA", "Psite", "Value"])

    # --- Residuals ---
    residuals = mRNA_mat - predictions
    df_residuals = pd.DataFrame(residuals, columns=[f"x{j+1}" for j in range(residuals.shape[1])])
    df_residuals.insert(0, "TF", mRNA_ids)

    # --- Observed ---
    df_observed = pd.DataFrame(mRNA_mat, columns=[f"x{j+1}" for j in range(mRNA_mat.shape[1])])
    df_observed.insert(0, "TF", mRNA_ids)

    # --- Estimated ---
    df_estimated = pd.DataFrame(predictions, columns=[f"x{j+1}" for j in range(predictions.shape[1])])
    df_estimated.insert(0, "TF", mRNA_ids)

    # --- Optimization Results ---
    y_true = mRNA_mat.flatten()
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

    # Write to Excel
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_alpha.to_excel(writer, sheet_name="Alpha Values", index=False)
        df_beta.to_excel(writer, sheet_name="Beta Values", index=False)
        df_residuals.to_excel(writer, sheet_name="Residuals", index=False)
        df_observed.to_excel(writer, sheet_name="Observed", index=False)
        df_estimated.to_excel(writer, sheet_name="Estimated", index=False)
        df_metrics.to_excel(writer, sheet_name="Optimization Results", index=False)

# -------------------------------
# Multi-Objective Problem Definition
# -------------------------------
class TFOptimizationMultiObjectiveProblem(Problem):
    def __init__(self, n_var, n_mRNA, n_TF, n_reg, n_psite_max, n_alpha,
                 mRNA_mat, regulators, protein_mat, psite_tensor, T_use,
                 beta_start_indices, num_psites, no_psite_tf, xl=None, xu=None,
                 **kwargs):
        # Three objectives: f1 (error), f2 (alpha violation), f3 (beta violation).
        super().__init__(n_var=n_var, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.n_mRNA = n_mRNA
        self.n_TF = n_TF
        self.n_reg = n_reg
        self.n_psite_max = n_psite_max
        self.n_alpha = n_alpha
        self.mRNA_mat = mRNA_mat
        self.regulators = regulators
        self.protein_mat = protein_mat
        self.psite_tensor = psite_tensor
        self.T_use = T_use
        self.beta_start_indices = beta_start_indices
        self.num_psites = num_psites
        self.no_psite_tf = no_psite_tf

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.empty((n_pop, 3))
        n_alpha = self.n_alpha
        for i in range(n_pop):
            xi = X[i]
            f1 = objective_(xi, self.mRNA_mat, self.regulators, self.protein_mat,
                                       self.psite_tensor, self.n_reg, self.T_use, self.n_mRNA, self.beta_start_indices, self.num_psites)
            f2 = 0.0
            for m in range(self.n_mRNA):
                s = 0.0
                for r in range(self.n_reg):
                    s += xi[m * self.n_reg + r]
                f2 += (s - 1.0) ** 2
            f3 = 0.0
            for tf in range(self.n_TF):
                start = n_alpha + self.beta_start_indices[tf]
                length = 1 + self.num_psites[tf]
                beta_vec = xi[start : start + length]
                f3 += (np.sum(beta_vec) - 1.0) ** 2
                if self.no_psite_tf[tf]:
                    for q in range(1, length):
                        f3 += beta_vec[q] ** 2
            F[i, 0] = f1
            F[i, 1] = f2
            F[i, 2] = f3
        out["F"] = F

# -------------------------------
# Main Optimization Routine
# -------------------------------
def main():
    # Load raw data.
    mRNA_ids, mRNA_mat, mRNA_time_cols = load_mRNA_data("../kinopt/data/input3.csv")
    TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols = load_TF_data(
        "../kinopt/data/input1_msgauss.csv")
    # Reverse interpretation: mRNA are in Source, TFs in Target.
    reg_map = load_regulation("../kinopt/data/input4_reduced.csv")

    # Print time points (in minutes) for clarity.
    time_points = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    print("Time points (minutes):", time_points)

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

    # Filter TFs to only those that regulate some mRNA.
    TF_ids_filtered = [tf for tf in TF_ids if tf in relevant_TFs]
    TF_ids = TF_ids_filtered
    protein_dict = {tf: protein_dict[tf] for tf in TF_ids}
    psite_dict = {tf: psite_dict[tf] for tf in TF_ids}
    psite_labels_dict = {tf: psite_labels_dict[tf] for tf in TF_ids}

    # Use common number of time points.
    T_use = min(mRNA_mat.shape[1], len(TF_time_cols))
    mRNA_mat = mRNA_mat[:, :T_use]

    # Build fixed arrays.
    mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite_max, psite_labels_arr, num_psites = \
        build_fixed_arrays(mRNA_ids, mRNA_mat, TF_ids, protein_dict, psite_dict, psite_labels_dict, reg_map)
    n_mRNA = mRNA_mat.shape[0]
    n_TF = protein_mat.shape[0]
    print(f"n_mRNA: {n_mRNA}, n_TF: {n_TF}, T_use: {T_use}, n_reg: {n_reg}, n_psite_max: {n_psite_max}")

    # Create boolean array for TFs with no PSite data.
    no_psite_tf = np.array([ (num_psites[i] == 0) or all(label == "" for label in psite_labels_arr[i])
                             for i in range(n_TF) ])

    # Compute cumulative starting indices for beta parameters for each TF.
    beta_start_indices = np.zeros(n_TF, dtype=np.int32)
    cum = 0
    for i in range(n_TF):
        beta_start_indices[i] = cum
        cum += 1 + num_psites[i]
    n_beta_total = cum

    # Build initial guess vector x0.
    n_alpha = n_mRNA * n_reg
    x0_alpha = np.full(n_alpha, 1.0 / n_reg)
    x0_beta_list = []
    for i in range(n_TF):
        if no_psite_tf[i]:
            x0_beta_list.extend([1.0])  # Only protein beta.
        else:
            length = 1 + num_psites[i]
            x0_beta_list.extend([1.0 / length] * length)
    x0_beta = np.array(x0_beta_list)
    x0 = np.concatenate([x0_alpha, x0_beta])
    total_dim = len(x0)
    print(f"Total parameter dimension: {total_dim}")

    # Define lower and upper bounds.
    xl = np.concatenate([np.zeros(n_alpha), -4.0 * np.ones(n_beta_total)])
    xu = np.concatenate([np.ones(n_alpha), 4.0 * np.ones(n_beta_total)])

    # 1) Determine the number of threads via lscpu
    n_threads_cmd = "lscpu -p | grep -v '^#' | wc -l"
    n_threads = int(subprocess.check_output(n_threads_cmd, shell=True).decode().strip())

    # 2) Create a thread pool and a parallelization runner
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # Create the multi-objective problem instance.
    problem = TFOptimizationMultiObjectiveProblem(
        n_var=total_dim, n_mRNA=n_mRNA, n_TF=n_TF, n_reg=n_reg,
        n_alpha=n_alpha, mRNA_mat=mRNA_mat, regulators=regulators,
        protein_mat=protein_mat, psite_tensor=psite_tensor, T_use=T_use,
        no_psite_tf=no_psite_tf, xl=xl, xu=xu, beta_start_indices=beta_start_indices,
        num_psites=num_psites, n_psite_max=n_psite_max,
        elementwise_runner=runner
    )

    # Common settings for the algorithms:
    pop_size = 500
    crossover = TwoPointCrossover(prob=0.9)
    mutation = PolynomialMutation(prob=1.0 / 120, eta=20)  # Adjust mutation probability as 1/n_var.
    eliminate_duplicates = True

    # NSGA2 settings.
    nsga2 = NSGA2(
        pop_size=pop_size,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=eliminate_duplicates
    )

    # SMSEMOA settings.
    smsemoa = SMSEMOA(
        pop_size=pop_size,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=eliminate_duplicates
    )

    # AGEMOEA settings.
    agemoea = AGEMOEA(
        pop_size=pop_size,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=eliminate_duplicates
    )

    termination = DefaultMultiObjectiveTermination()

    res = pymoo_minimize(problem=problem,
                         algorithm=nsga2,
                         termination=termination,
                         seed=1,
                         verbose=True)

    if res.X is None:
        print("No feasible solution found by pymoo. Exiting.")
        return

    pool.close()

    print("Optimization Result:")
    print(res)

    # Extract the Pareto front and choose a best solution by weighting the objectives.
    pareto_front = np.array([ind.F for ind in res.pop])
    weights = np.array([1.0, 1.0, 1.0])
    scores = pareto_front[:, 0] + weights[1] * np.abs(pareto_front[:, 1]) + weights[2] * np.abs(pareto_front[:, 2])
    best_index = np.argmin(scores)
    best_solution = res.pop[best_index]
    best_objectives = pareto_front[best_index]
    final_x = best_solution.X
    print("\n--- Best Solution ---")
    print(f"Objective Values (F): {best_objectives}")

    final_alpha = final_x[:n_alpha].reshape((n_mRNA, n_reg))
    final_beta = []
    # Extract beta parameters per TF using beta_start_indices.
    for i in range(n_TF):
        start = beta_start_indices[i]
        length = 1 + num_psites[i]
        final_beta.append(final_x[n_alpha + start : n_alpha + start + length])
    final_beta = np.array(final_beta, dtype=object)  # Use object array to hold variable-length vectors

    # Build mapping for α.
    alpha_mapping = {}
    for i, mrna in enumerate(mRNA_ids):
        actual_tfs = [tf for tf in reg_map[mrna] if tf in TF_ids]  # Only valid TFs
        alpha_mapping[mrna] = {}
        for j, tf in enumerate(actual_tfs):
            alpha_mapping[mrna][tf] = final_alpha[i, j]

    # Display α mapping in the same structure as the Excel output.
    print("\nMapping of TFs to mRNAs (α values):")
    for mrna, mapping in alpha_mapping.items():
        print(f"{mrna}:")
        for tf, a_val in mapping.items():
            print(f"   {tf}: {a_val:.4f}")

    # Build mapping for β.
    beta_mapping = {}
    for idx, tf in enumerate(TF_ids):
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

    # Compute predictions.
    predictions = compute_predictions(final_x, regulators, protein_mat, psite_tensor, n_reg, T_use, n_mRNA, beta_start_indices, num_psites)
    # Plot observed vs. estimated mRNA time series with overlaid TF protein signals.
    plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, mRNA_time_cols, regulators, protein_mat, TF_ids,
                               num_targets=14)
    save_results_to_excel(mRNA_ids, TF_ids, final_alpha, final_beta, psite_labels_arr, mRNA_mat, predictions, best_objectives, reg_map)


if __name__ == "__main__":
    main()