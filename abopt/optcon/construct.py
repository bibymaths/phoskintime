import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import csr_matrix
from typing import Tuple
from numpy.typing import NDArray

def _build_P_initial(full_df, interact_df):
    time = [f'x{i}' for i in range(1, 15)]
    P_initial = {}
    P_list = []
    for _, row in interact_df.iterrows():
        gene, psite, kinases = row['GeneID'], row['Psite'], row['Kinase']
        kinases = [k.strip() for k in kinases]
        ts = full_df[(full_df['GeneID'] == gene) & (full_df['Psite'] == psite)][time].values.flatten()
        if ts.size == 0:
            ts = np.ones(len(time))
        P_list.append(ts)
        P_initial[(gene, psite)] = {'Kinases': kinases, 'TimeSeries': ts}
    return P_initial, np.array(P_list)

def _build_K_data(full_df, interact_df, estimate_missing):
    time = [f'x{i}' for i in range(1, 15)]
    K_index = {}
    K_list = []
    beta_counts = {}
    synthetic_counter = 1
    for kinase in interact_df['Kinase'].explode().unique():
        kinase_df = full_df[full_df['GeneID'] == kinase]
        if not kinase_df.empty:
            for _, row in kinase_df.iterrows():
                psite = row['Psite']
                ts = np.array(row[time].values, dtype=np.float64)
                idx = len(K_list)
                K_list.append(ts)
                K_index.setdefault(kinase, []).append((psite, ts))
                beta_counts[idx] = 1
        elif estimate_missing:
            K_index.setdefault(kinase, [])
            for _ in range(1):  # default one synthetic psite
                synthetic_label = f"P{synthetic_counter}"
                synthetic_counter += 1
                synthetic_ts = np.empty(len(time))
                K_list.append(synthetic_ts)
                K_index[kinase].append((synthetic_label, synthetic_ts))
                beta_counts[len(K_list) - 1] = 1
    return K_index, np.array(K_list), beta_counts


def _convert_to_sparse(K_array):
    K_sparse = csr_matrix(K_array)
    return K_sparse, K_sparse.data, K_sparse.indices, K_sparse.indptr


def _precompute_mappings(P_initial, K_index):
    # Unique kinases from P_initial
    unique_kinases = sorted(list({k for key in P_initial for k in P_initial[key]['Kinases']}))
    kinase_to_idx = {k: i for i, k in enumerate(unique_kinases)}
    n_gene = len(P_initial)
    gene_kinase_counts = np.empty(n_gene, dtype=np.int32)
    gene_alpha_starts = np.empty(n_gene, dtype=np.int32)
    gene_kinase_idx_list = []
    cum = 0
    for i, key in enumerate(P_initial.keys()):
        kinases = P_initial[key]['Kinases']
        count = len(kinases)
        gene_kinase_counts[i] = count
        gene_alpha_starts[i] = cum
        for k in kinases:
            gene_kinase_idx_list.append(kinase_to_idx[k])
        cum += count
    gene_kinase_idx = np.array(gene_kinase_idx_list, dtype=np.int32)
    total_alpha = cum

    n_kinase = len(unique_kinases)
    kinase_beta_counts = np.empty(n_kinase, dtype=np.int32)
    kinase_beta_starts = np.empty(n_kinase, dtype=np.int32)
    cum = 0
    for i, k in enumerate(unique_kinases):
        cnt = len(K_index[k]) if k in K_index else 0
        kinase_beta_counts[i] = cnt
        kinase_beta_starts[i] = cum
        cum += cnt
    return unique_kinases, gene_kinase_counts, gene_alpha_starts, gene_kinase_idx, total_alpha, kinase_beta_counts, kinase_beta_starts

def _init_parameters(
    total_alpha: int,
    lb: float,
    ub: float,
    kinase_beta_counts: list[int]
) -> Tuple[NDArray[np.float64], list[tuple[float, float]]]:
    """
    ...
    """
    n_beta = int(sum(kinase_beta_counts))
    bounds = [(0.0, 1.0)] * total_alpha + [(lb, ub)] * n_beta
    alpha_initial: NDArray[np.float64] = np.random.rand(total_alpha)
    beta_initial: NDArray[np.float64] = (
        np.random.rand(n_beta) if (lb == 0 and ub == 1)
        else np.random.uniform(lb, ub, size=n_beta)
    )
    params_initial: NDArray[np.float64] = np.concatenate([alpha_initial, beta_initial])
    return params_initial, bounds


def _compute_time_weights(P_array, loss_type):
    t_max = P_array.shape[1]
    P_dense = P_array.astype(np.float64)
    if loss_type == "weighted":
        time_weights = np.empty(t_max, dtype=np.float64)
        for t in range(t_max):
            var_t = np.var(P_dense[:, t])
            time_weights[t] = 1.0 / (var_t + 1e-8)
    else:
        time_weights = np.ones(t_max, dtype=np.float64)
    return t_max, P_dense, time_weights

def _eq_constraint(s, c):
    def f(p):
        return np.sum(p[s : s + c]) - 1
    return f

def _build_constraints(opt_method, gene_kinase_counts, unique_kinases, total_alpha, kinase_beta_counts, n_params):
    if opt_method == "trust-constr":
        n_alpha = len(gene_kinase_counts)
        n_beta = len(unique_kinases)
        total = n_alpha + n_beta
        A = np.zeros((total, n_params))
        row = 0
        alpha_start = 0
        for count in gene_kinase_counts:
            A[row, alpha_start:alpha_start+count] = 1.0
            row += 1
            alpha_start += count
        beta_start = total_alpha
        for k in range(len(unique_kinases)):
            cnt = kinase_beta_counts[k]
            A[row, beta_start:beta_start+cnt] = 1.0
            row += 1
            beta_start += cnt
        return [LinearConstraint(A, lb=1, ub=1)]
    else:
        cons = []
        alpha_start = 0
        for count in gene_kinase_counts:
            cons.append({
                'type': 'eq',
                'fun': _eq_constraint(alpha_start, count)
            })
            alpha_start += count

        beta_start = total_alpha
        for bc in kinase_beta_counts:
            cons.append({
                'type': 'eq',
                'fun': _eq_constraint(beta_start, bc)
            })
            beta_start += bc

        return cons