import numpy as np
import multiprocessing as mp

import pandas as pd
from scipy import sparse


def _build_single_W(args):
    p, interactions, sites_i, k2i, n_kinases = args

    sub = interactions[interactions["protein"] == p]
    site_map = {s: r for r, s in enumerate(sites_i)}
    rows, cols = [], []

    for _, r in sub.iterrows():
        if r["psite"] in site_map and r["kinase"] in k2i:
            rows.append(site_map[r["psite"]])
            cols.append(k2i[r["kinase"]])

    data = np.ones(len(rows), float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sites_i), n_kinases))


def build_W_parallel(interactions: pd.DataFrame, idx, n_cores=4) -> sparse.csr_matrix:
    print(f"[Model] Building W matrices in parallel using {n_cores} cores...")

    tasks = [
        (p, interactions, idx.sites[i], idx.k2i, len(idx.kinases))
        for i, p in enumerate(idx.proteins)
    ]

    if n_cores <= 1:
        W_list = list(map(_build_single_W, tasks))
    else:
        with mp.Pool(n_cores) as pool:
            W_list = pool.map(_build_single_W, tasks)

    print("[Model] Stacking Global W matrix...")
    return sparse.vstack(W_list).tocsr()


def build_tf_matrix(tf_net, idx):
    rows, cols = [], []
    for _, r in tf_net.iterrows():
        if r["tf"] in idx.p2i and r["target"] in idx.p2i:
            rows.append(idx.p2i[r["target"]])
            cols.append(idx.p2i[r["tf"]])
    data = np.ones(len(rows), float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(idx.N, idx.N))
