"""Build transcription-factor and kinase-to-site matrices from interaction tables and index maps; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config."""

import re
import multiprocessing as mp

import pandas as pd
from scipy import sparse
from config.config import setup_logger
from networkmodel.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def site_key(site: str) -> int:
    """Normalize a phosphorylation site label
    
    Args:
        site: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
    """
    m = re.search(r"\d+", site)
    if m is None:
        raise ValueError(f"Invalid site format: {site}")
    return int(m.group())


def _build_single_W(args):
    """Handle internal build single W"""
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
    """Build kinase-to-site weight matrices in parallel
    
    Args:
        interactions: Input value used by this routine.
        idx: Input value used by this routine.
        n_cores: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
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
    """Build the transcription-factor regulatory matrix
    
    Args:
        tf_net: Input value used by this routine.
        idx: Input value used by this routine.
        tf_beta_map: Input value used by this routine.
        kin_beta_map: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
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
