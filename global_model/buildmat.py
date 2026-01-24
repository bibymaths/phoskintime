"""
Network Matrix Construction Module.

This script handles the construction of the two primary connectivity matrices
used in the global simulation:
1.  **W Matrix (Kinase-Substrate):** A sparse matrix defining the phosphorylation
    rates from kinases to specific phosphosites. It is built in parallel to handle
    large interactomes.
2.  **TF Matrix (Transcription Factor-Gene):** A sparse matrix defining the
    transcriptional regulation strengths. It includes logic to handle "proxy"
    TFs (orphans mapped to kinases).
"""

import re
import multiprocessing as mp

import pandas as pd
from scipy import sparse
from config.config import setup_logger
from global_model.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def site_key(site: str) -> int:
    """
    Extracts the numerical position from a phosphosite string for sorting.

    Args:
        site (str): e.g., "S473" or "Y1068".

    Returns:
        int: The integer position (e.g., 473).

    Raises:
        ValueError: If no digits are found in the string.
    """
    m = re.search(r"\d+", site)
    if m is None:
        raise ValueError(f"Invalid site format: {site}")
    return int(m.group())


def _build_single_W(args):
    """
    Worker function to build the sub-matrix W for a single protein.

    Constructs a sparse matrix of shape $(N_{sites}, N_{kinases})$ for the
    specific protein $p$. The values are determined by the interaction strength $\alpha$.

    Args:
        args (tuple): (protein_name, interactions_df, sites_list, kinase_map, n_kinases)

    Returns:
        scipy.sparse.csr_matrix: The local W matrix for the given protein.
    """
    p, interactions, sites_i, k2i, n_kinases = args

    # Sort sites for consistent row ordering
    sites_i.sort(key=site_key)

    # Filter interactions for this protein
    sub = interactions[interactions["protein"] == p]

    # Map site name -> local row index
    site_map = {s: r for r, s in enumerate(sites_i)}

    rows, cols, data = [], [], []

    for _, r in sub.iterrows():
        # Only add if site and kinase are valid in our index
        if r["psite"] in site_map and r["kinase"] in k2i:
            rows.append(site_map[r["psite"]])
            cols.append(k2i[r["kinase"]])

            # --- CRITICAL FIX: Use the 'alpha' column ---
            # Default to 1.0 if missing, but it should be there from load_data
            # Weight represents the base catalytic efficiency or affinity.
            weight = float(r.get("alpha", 1.0))
            data.append(weight)

    # Use the 'data' list instead of np.ones
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sites_i), n_kinases))


def build_W_parallel(interactions: pd.DataFrame, idx, n_cores=4) -> sparse.csr_matrix:
    """
    Constructs the Global Kinase-Substrate Matrix W using parallel processing.



    The global matrix is a vertical stack of local matrices:
    $$
    W_{global} = \begin{bmatrix} W_{protein_1} \\ W_{protein_2} \\ \vdots \end{bmatrix}
    $$

    Args:
        interactions (pd.DataFrame): Dataframe containing 'protein', 'psite', 'kinase', and 'alpha'.
        idx (IndexMap): Object containing mappings (sites, k2i, proteins).
        n_cores (int): Number of parallel processes to use.

    Returns:
        sparse.csr_matrix: The global interaction matrix.
    """
    logger.info(f"[Model] Building W matrices in parallel using {n_cores} cores...")

    # Prepare tasks
    # interactions df now contains the 'alpha' column from load_data
    tasks = [
        (p, interactions, idx.sites[i], idx.k2i, len(idx.kinases))
        for i, p in enumerate(idx.proteins)
    ]

    if n_cores <= 1:
        W_list = list(map(_build_single_W, tasks))
    else:
        with mp.Pool(n_cores) as pool:
            W_list = pool.map(_build_single_W, tasks)

    logger.info("[Model] Stacking Global W matrix...")
    return sparse.vstack(W_list).tocsr()


def build_tf_matrix(tf_net, idx, tf_beta_map=None, kin_beta_map=None):
    """
    Constructs the Transcription Factor Regulation Matrix.

    This matrix defines the influence of TFs on target genes.
    The edge weight is calculated as:
    $$ W_{ij} = \alpha_{ij} \times \beta_{tf} $$
    where $\alpha$ is the edge strength and $\beta$ is the activity multiplier.

    **Proxy Logic:**
    If a TF is mapped as a "proxy" (i.e., an orphan TF whose activity is inferred
    from a kinase), we use the kinase's beta value (`kin_beta_map`) instead of
    the TF's beta.

    Args:
        tf_net (pd.DataFrame): TF-Target interactions with 'tf', 'target', and 'alpha'.
        idx (IndexMap): Object containing mappings (p2i, proxy_map).
        tf_beta_map (dict, optional): Multipliers for standard TFs.
        kin_beta_map (dict, optional): Multipliers for kinases (used for proxies).

    Returns:
        sparse.csr_matrix: An $(N, N)$ matrix where rows=targets, cols=TFs.
    """
    if tf_beta_map is None: tf_beta_map = {}
    if kin_beta_map is None: kin_beta_map = {}

    rows, cols, data = [], [], []

    for _, r in tf_net.iterrows():
        tf = r["tf"]
        target = r["target"]

        if tf in idx.p2i and target in idx.p2i:
            rows.append(idx.p2i[target])
            cols.append(idx.p2i[tf])

            # Get the base edge strength (Alpha)
            alpha = float(r.get("alpha", 1.0))

            # --- Proxy-Aware Beta Selection ---
            # Check if this TF name is a redirected Orphan in the index
            if hasattr(idx, 'proxy_map') and tf in idx.proxy_map:
                proxy_kinase = idx.proxy_map[tf]
                # Use the Kinase multiplier (c_k) as the activity weight
                beta = float(kin_beta_map.get(proxy_kinase, 1.0))
            else:
                # Use standard TF intrinsic beta
                beta = float(tf_beta_map.get(tf, 1.0))

            # Apply absolute weight to ensure positive synthesis contributions
            # (Repression is handled by the sign of tf_scale in the RHS)
            weight = alpha * beta  # abs(beta)
            data.append(weight)

    return sparse.csr_matrix((data, (rows, cols)), shape=(idx.N, idx.N))