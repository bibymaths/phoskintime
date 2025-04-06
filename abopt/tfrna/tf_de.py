
import pandas as pd
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination


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
    n_mRNA, T = mRNA_mat.shape

    # Map TF_id to index.
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

    # Build regulators array (n_mRNA x n_reg), pad with index 0.
    regulators = np.zeros((n_mRNA, n_reg), dtype=np.int32)
    for i, regs in enumerate(reg_list):
        for j in range(n_reg):
            if j < len(regs):
                regulators[i, j] = TF_index.get(regs[j], 0)
            else:
                regulators[i, j] = 0

    # Build protein_mat.
    protein_mat = np.zeros((n_TF, T), dtype=np.float64)
    for tf, idx in TF_index.items():
        if protein_dict.get(tf) is not None:
            protein_mat[idx, :] = protein_dict[tf][:T]
        else:
            protein_mat[idx, :] = np.zeros(T)

    # Determine maximum number of PSites among TFs.
    max_psite = 0
    for tf in TF_ids:
        n = len(psite_dict.get(tf, []))
        max_psite = max(max_psite, n)
    n_psite = max_psite  # If no TF has PSite data, n_psite==0.

    # Build psite_tensor and psite_labels_arr.
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
# Numba-Accelerated Objective (f1)
# -------------------------------
@njit(parallel=True)
def objective_numba_fixed(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA, n_TF):
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
# Additional Objective Functions
# -------------------------------
def alpha_objective(x, n_mRNA, n_reg):
    # f2: sum over mRNA of (sum(alpha) - 1)^2.
    f2 = 0.0
    for i in range(n_mRNA):
        s = 0.0
        for r in range(n_reg):
            s += x[i * n_reg + r]
        f2 += (s - 1.0) ** 2
    return f2

def beta_objective(x, n_alpha, n_TF, n_psite, no_psite_tf):
    # f3: for each TF, add (sum(beta) - 1)^2.
    # For TF with no PSite, also add penalty for extra beta's: sum_{q=1}^{n_psite} (beta_q)^2.
    f3 = 0.0
    for tf in range(n_TF):
        start = n_alpha + tf * (1 + n_psite)
        beta_vec = x[start : start + (1 + n_psite)]
        f3 += (np.sum(beta_vec) - 1.0) ** 2
        if no_psite_tf[tf]:
            for q in range(1, 1 + n_psite):
                f3 += beta_vec[q] ** 2
    return f3

# -------------------------------
# Plotting Functions
# -------------------------------
def compute_predictions(x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA):
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

def plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, time_points, regulators, protein_mat, TF_ids, num_targets=5):
    T = len(time_points)
    time_vals = np.array([4, 8, 15, 30, 60, 120, 240, 480, 960])
    num_targets = min(num_targets, predictions.shape[0])
    plt.figure(figsize=(10, num_targets * 3))
    for i in range(num_targets):
        plt.subplot(num_targets, 1, i + 1)
        plt.plot(time_vals, mRNA_mat[i, :], 'o-', label='Observed')
        plt.plot(time_vals, predictions[i, :], 's--', label='Estimated')
        plotted_tfs = set()
        for r in regulators[i, :]:
            tf_name = TF_ids[r]
            if tf_name not in plotted_tfs:
                protein_signal = protein_mat[r, :T]
                plt.plot(time_vals, protein_signal, ':', label=f"TF {tf_name}")
                plotted_tfs.add(tf_name)
        plt.title(f"mRNA: {mRNA_ids[i]}")
        plt.xlabel("Time Point")
        plt.ylabel("Expression")
        plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Multi-Objective Problem Definition
# -------------------------------
class TFOptimizationMultiObjectiveProblem(Problem):
    def __init__(self, n_var, n_mRNA, n_TF, n_reg, n_psite, n_alpha,
                 mRNA_mat, regulators, protein_mat, psite_tensor, T_use, no_psite_tf, xl=None, xu=None):
        # Three objectives: f1 (error), f2 (alpha constraint violation), f3 (beta constraint violation).
        super().__init__(n_var=n_var, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.n_mRNA = n_mRNA
        self.n_TF = n_TF
        self.n_reg = n_reg
        self.n_psite = n_psite
        self.n_alpha = n_alpha
        self.mRNA_mat = mRNA_mat
        self.regulators = regulators
        self.protein_mat = protein_mat
        self.psite_tensor = psite_tensor
        self.T_use = T_use
        self.no_psite_tf = no_psite_tf

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.empty((n_pop, 3))
        n_alpha = self.n_mRNA * self.n_reg
        for i in range(n_pop):
            xi = X[i]
            # Objective 1: Prediction error.
            f1 = objective_numba_fixed(xi, self.mRNA_mat, self.regulators, self.protein_mat,
                                       self.psite_tensor, self.n_reg, self.n_psite, self.T_use, self.n_mRNA, self.n_TF)
            # Objective 2: Alpha constraint violation.
            f2 = 0.0
            for m in range(self.n_mRNA):
                s = 0.0
                for r in range(self.n_reg):
                    s += xi[m * self.n_reg + r]
                f2 += (s - 1.0) ** 2
            # Objective 3: Beta constraint violation.
            f3 = 0.0
            for tf in range(self.n_TF):
                start = n_alpha + tf * (1 + self.n_psite)
                beta_vec = xi[start : start + (1 + self.n_psite)]
                f3 += (np.sum(beta_vec) - 1.0) ** 2
                if self.no_psite_tf[tf]:
                    for q in range(1, 1 + self.n_psite):
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
    mRNA_ids, mRNA_mat, mRNA_time_cols = load_mRNA_data("../data/input3.csv")
    TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols = load_TF_data("../data/input1_msgauss.csv")
    reg_map = load_regulation("../data/input4_reduced.csv")

    # Print time points as a NumPy array (in minutes).
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

    # Use common number of time points (T_use)
    T_use = min(mRNA_mat.shape[1], len(TF_time_cols))
    mRNA_mat = mRNA_mat[:, :T_use]

    # Build fixed arrays.
    mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, psite_labels_arr = \
        build_fixed_arrays(mRNA_ids, mRNA_mat, TF_ids, protein_dict, psite_dict, psite_labels_dict, reg_map)
    n_mRNA = mRNA_mat.shape[0]
    n_TF = protein_mat.shape[0]
    print(f"n_mRNA: {n_mRNA}, n_TF: {n_TF}, T_use: {T_use}, n_reg: {n_reg}, n_psite: {n_psite}")

    # Create boolean array for TFs with no PSite data.
    no_psite_tf = np.array([ (n_psite == 0) or all(label == "" for label in psite_labels_arr[idx])
                             for idx in range(n_TF) ])

    # Build initial guess vector x0.
    n_alpha = n_mRNA * n_reg
    x0_alpha = np.full(n_alpha, 1.0 / n_reg)
    n_beta = n_TF * (1 + n_psite)
    # For each TF with no PSite data, initialize beta as [1, 0, 0, ...].
    x0_beta_list = []
    for i in range(n_TF):
        if no_psite_tf[i]:
            x0_beta_list.extend([1.0] + [0.0] * n_psite)
        else:
            x0_beta_list.extend([1.0 / (1 + n_psite)] * (1 + n_psite))
    x0_beta = np.array(x0_beta_list)
    x0 = np.concatenate([x0_alpha, x0_beta])
    total_dim = len(x0)
    print(f"Total parameter dimension: {total_dim}")

    # Define bounds.
    xl = np.concatenate([np.zeros(n_alpha), -4.0 * np.ones(n_beta)])
    xu = np.concatenate([np.ones(n_alpha), 4.0 * np.ones(n_beta)])

    # Create the multi-objective problem instance.
    problem = TFOptimizationMultiObjectiveProblem(
        n_var=total_dim, n_mRNA=n_mRNA, n_TF=n_TF, n_reg=n_reg, n_psite=n_psite,
        n_alpha=n_alpha, mRNA_mat=mRNA_mat, regulators=regulators,
        protein_mat=protein_mat, psite_tensor=psite_tensor, T_use=T_use,
        no_psite_tf=no_psite_tf, xl=xl, xu=xu
    )

    # Use NSGA2 from pymoo.
    algorithm = NSGA2(
        pop_size=500,
        crossover=TwoPointCrossover(),
        eliminate_duplicates=True
    )
    termination = DefaultMultiObjectiveTermination()

    res = pymoo_minimize(problem=problem,
                         algorithm=algorithm,
                         termination=termination,
                         seed=1,
                         verbose=True)

    if res.X is None:
        print("No feasible solution found by pymoo. Exiting.")
        return

    # Extract all objective values (F) from the Pareto front
    pareto_front = np.array([ind.F for ind in res.pop])
    # Define weights for the objectives (e.g., prioritize F[0])
    weights = np.array([1.0, 1.0, 1.0])  # Adjust weights as needed
    scores = pareto_front[:, 0] + weights[1] * np.abs(pareto_front[:, 1]) + weights[2] * np.abs(pareto_front[:, 2])
    # Find the index of the best solution (smallest score)
    best_index = np.argmin(scores)
    # Extract the corresponding solution and objectives
    best_solution = res.pop[best_index]
    best_objectives = pareto_front[best_index]
    final_x = best_solution.X
    # Display the selected solution and objectives
    print("\n--- Best Solution ---")
    print(f"Objective Values (F): {best_objectives}")

    final_alpha = final_x[:n_alpha].reshape((n_mRNA, n_reg))
    final_beta = final_x[n_alpha:].reshape((n_TF, 1 + n_psite))

    # Build mapping for α.
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

    # Build mapping for β.
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

    # Compute predictions.
    predictions = compute_predictions(final_x, mRNA_mat, regulators, protein_mat, psite_tensor, n_reg, n_psite, T_use, n_mRNA)
    # Plot observed vs. estimated mRNA time series with overlaid TF protein signals.
    plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, mRNA_time_cols, regulators, protein_mat, TF_ids, num_targets=15)

if __name__ == "__main__":
    main()
