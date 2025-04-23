import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import csr_matrix
from typing import Tuple
from numpy.typing import NDArray

from kinopt.local.config.constants import INPUT2, INPUT1 
 
from kinopt.local.config.logconf import setup_logger
logger = setup_logger()


def _build_P_initial(full_df, interact_df):
    """
    Build the initial protein-kinase mapping and time series data.

    :param full_df:
    :param interact_df:
    :return: Protein - Kinase mapping, time series data
    """
    # Extract time series columns
    time = [f'x{i}' for i in range(1, 15)]
    # Initialize the dictionary to hold time series data
    P_initial = {}
    P_list = []
    # Iterate through the interaction dataframe
    for _, row in interact_df.iterrows():
        # Extract gene, phosphorylation site, and kinases
        gene, psite, kinases = row['GeneID'], row['Psite'], row['Kinase']
        # If the gene is not in the full dataframe, skip it
        kinases = [k.strip() for k in kinases]
        # Grab the time series data for the gene and phosphorylation site
        # Check in full_df if that (gene, psite) combination exists and get time series
        match = full_df[(full_df['GeneID'] == gene) & (full_df['Psite'] == psite)]
        if not match.empty:
            ts = match.iloc[0][time].values.astype(np.float64)
        # Append the time series data to the list
        P_list.append(ts)
        # Store the gene, phosphorylation site, and kinases in the dictionary
        P_initial[(gene, psite)] = {'Kinases': kinases, 'TimeSeries': ts}
    return P_initial, np.array(P_list)

def _build_K_data(full_df, interact_df, estimate_missing):
    """
    Build the kinase data for optimization.

    :param full_df: DataFrame containing full data
    :param interact_df: DataFrame containing interactions
    :param estimate_missing: Boolean flag to estimate missing data
    :return: Kinase index, kinase data array, beta counts
    """
    # Extract time series columns
    time = [f'x{i}' for i in range(1, 15)]

    # Initialize the dictionary to hold kinase data
    K_index = {}
    K_list = []
    # Initialize the list to hold beta counts
    beta_counts = {}

    synthetic_counter = 1
    synthetic_rows = []

    # Iterate through the interaction dataframe
    for kinase in interact_df['Kinase'].explode().unique():
        # Extract the gene and phosphorylation site for the current kinase
        kinase_df = full_df[full_df['GeneID'] == kinase]
        # If the kinase is not in the full dataframe, skip it
        if not kinase_df.empty:
            # Extract the time series data for the kinase
            for _, row in kinase_df.iterrows():
                # Extract the gene, phosphorylation site, and time series data
                psite = row['Psite']
                ts = np.array(row[time].values, dtype=np.float64)
                # Store index and time series data in the dictionary
                idx = len(K_list)
                K_list.append(ts)
                # Store the kinase, phosphorylation site, and time series data
                K_index.setdefault(kinase, []).append((psite, ts))
                # Update the beta counts
                beta_counts[idx] = 1
        # If the kinase is not in the full dataframe and we are estimating missing data
        elif estimate_missing:
            # Create a synthetic label for the kinase
            synthetic_label = f"P{synthetic_counter}"
            synthetic_counter += 1
            # Get protein time series for this kinase where 'Psite' is empty or NaN
            protein_level_df = full_df[(full_df['GeneID'] == kinase) & (full_df['Psite'].isna())]
            if not protein_level_df.empty:
                synthetic_ts = np.array(protein_level_df.iloc[0][time].values, dtype=np.float64)
            idx = len(K_list)
            K_list.append(synthetic_ts)
            K_index.setdefault(kinase, []).append((synthetic_label, synthetic_ts))
            beta_counts[idx] = 1
            synthetic_rows.append(idx)

    # Finalize K_array
    K_array = np.array(K_list)

    return K_index, K_array, beta_counts

def _convert_to_sparse(K_array):
    """
    Function to convert a dense array to sparse format.

    :param K_array: Kinase data array
    :return: Sparse matrix, data, indices, indptr
    """
    K_sparse = csr_matrix(K_array)
    return K_sparse, K_sparse.data, K_sparse.indices, K_sparse.indptr


def _precompute_mappings(P_initial, K_index):
    """
    Function to precompute mappings for optimization.

    :param P_initial: Initial protein-kinase mapping
    :param K_index: Kinase index
    :return: Unique kinases, gene-kinase counts, alpha starts, kinase indices, total alpha, beta counts, beta starts
    :rtype: tuple

    :note: This function precomputes the mappings for optimization.
    :note: It extracts unique kinases, gene-kinase counts, and initializes alpha and beta parameters.
    :note: The function also computes the total number of alpha parameters and the beta counts.
    :note: The function is used to prepare the data for optimization.
    """

    # Extract unique kinases and their indices
    unique_kinases = sorted(list({k for key in P_initial for k in P_initial[key]['Kinases']}))
    # Create a mapping from kinase to index
    kinase_to_idx = {k: i for i, k in enumerate(unique_kinases)}
    # Number of protein groups
    n_gene = len(P_initial)
    # Initialize arrays for gene-kinase counts and alpha starts
    gene_kinase_counts = np.empty(n_gene, dtype=np.int32)
    # Initialize arrays for gene-kinase indices and total alpha
    gene_alpha_starts = np.empty(n_gene, dtype=np.int32)
    # Initialize a list to hold gene-kinase indices
    gene_kinase_idx_list = []
    # Initialize a variable to hold the cumulative count of kinases
    cum = 0
    # Iterate through the initial protein-kinase mapping
    for i, key in enumerate(P_initial.keys()):
        # Extract the kinases for the current protein
        kinases = P_initial[key]['Kinases']
        # Map the kinases to their indices
        count = len(kinases)
        # Store the count of kinases for the current protein
        gene_kinase_counts[i] = count
        # Store the cumulative count of kinases for the current protein
        gene_alpha_starts[i] = cum
        # Store the indices of the kinases for the current protein
        for k in kinases:
            gene_kinase_idx_list.append(kinase_to_idx[k])
        # Update the cumulative count of kinases
        cum += count
    # Convert the list of gene-kinase indices to a numpy array
    gene_kinase_idx = np.array(gene_kinase_idx_list, dtype=np.int32)
    # Store the total number of alpha parameters
    total_alpha = cum

    # Initialize arrays for kinase beta counts and starts
    n_kinase = len(unique_kinases)
    # Initialize arrays for kinase beta counts and starts
    kinase_beta_counts = np.empty(n_kinase, dtype=np.int32)
    kinase_beta_starts = np.empty(n_kinase, dtype=np.int32)
    # Cumulative count of kinases - phosphorylation sites
    cum = 0
    # Iterate through the unique kinases
    for i, k in enumerate(unique_kinases):
        # Count the number of phosphorylation sites for the current kinase
        cnt = len(K_index[k]) if k in K_index else 0
        # Store the count of phosphorylation sites for the current kinase
        kinase_beta_counts[i] = cnt
        # Store the cumulative count of phosphorylation sites for the current kinase
        kinase_beta_starts[i] = cum
        # Store the indices of the phosphorylation sites for the current kinase
        cum += cnt

    return unique_kinases, gene_kinase_counts, gene_alpha_starts, gene_kinase_idx, total_alpha, kinase_beta_counts, kinase_beta_starts

def _init_parameters(
    total_alpha: int,
    lb: float,
    ub: float,
    kinase_beta_counts: list[int]
) -> Tuple[NDArray[np.float64], list[tuple[float, float]]]:
    """
    Function to initialize parameters for optimization.

    :param total_alpha: Total number of alpha parameters
    :param lb: Lower bound for beta parameters
    :param ub: Upper bound for beta parameters
    :param kinase_beta_counts: List of kinase beta counts
    :return: Initial parameters and bounds
    :rtype: tuple
    :note: This function initializes the parameters for optimization.
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
    """
    Function to compute time weights for optimization.

    :param P_array:
    :param loss_type:
    :return: t_max, P_dense, time_weights
    """
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
    """
    Function to create an equality constraint for optimization.

    :param s:
    :param c:
    :return: linear constraint sum
    """
    def f(p):
        """
        Function to compute the equality constraint for optimization.

        :param p:
        :return: Sum of parameters in the range [s, s+c] minus 1
        """
        return np.sum(p[s : s + c]) - 1
    return f

def _build_constraints(opt_method, gene_kinase_counts, unique_kinases, total_alpha, kinase_beta_counts, n_params):
    """
    Function to build constraints for optimization.

    :param opt_method:
    :param gene_kinase_counts:
    :param unique_kinases:
    :param total_alpha:
    :param kinase_beta_counts:
    :param n_params:
    :return: Constraints for optimization
    """
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
     
def load_geneid_to_psites(input1_path=INPUT1):
    geneid_psite_map = defaultdict(set)
    with open(input1_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            geneid = row['GeneID'].strip()
            psite = row['Psite'].strip()
            if geneid and psite:
                geneid_psite_map[geneid].add(psite)
    return geneid_psite_map

def get_unique_kinases(input2_path=INPUT2):
    kinases = set()
    with open(input2_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kinase_raw = row['Kinase'].strip()
            # Remove surrounding curly braces if present
            if kinase_raw.startswith("{") and kinase_raw.endswith("}"):
                kinase_raw = kinase_raw[1:-1]
            # Split if it's a list like {KIN1,KIN2}
            for k in kinase_raw.split(","):
                k = k.strip()
                if k:
                    kinases.add(k)
    return kinases

def check_kinases():
    geneid_to_psites = load_geneid_to_psites()
    kinases = get_unique_kinases()

    logger.info("--- Kinase Check in input1.csv ---")
    for kinase in sorted(kinases):
        if kinase in geneid_to_psites:
            psites = sorted(geneid_to_psites[kinase])
            logger.info(f"{kinase}: Found, Psites in input1 -> {psites}")
        else:
            logger.info(f"{kinase}: Missing")

    logger.info("--- Summary ---")
    found = sum(1 for k in kinases if k in geneid_to_psites)
    missing = len(kinases) - found
    logger.info(f"Total unique kinases: {len(kinases)}")
    logger.info(f"Found: {found}")
    logger.info(f"Missing: {missing}")