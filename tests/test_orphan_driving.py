"""N21: does the JAX RHS drive redirected orphan TFs differently from the legacy System.rhs convention?

Index redirects an orphan TF by overwriting p2i[orphan] with the proxy kinase's index; the JAX RHS therefore
assigns the proxy's driver to the proxy's own row, while System.rhs drove the orphan's own row by name.
The test builds a real Index and the real TF matrix and checks whether that difference can reach any
derivative: it only can if some TF-matrix entry reads or writes the orphan's own row/column.
"""
import numpy as np
import pandas as pd


def _index_with_orphan():
    from networkmodel.network import Index

    interactions = pd.DataFrame({"protein": ["A", "A", "K1"], "psite": ["S_1", "T_2", "Y_3"],
                                 "kinase": ["K1", "K2", "K2"]})
    # O has no sites in the signaling data -> orphan TF; it regulates kinase K1 -> proxy K1. O is also a target.
    tf = pd.DataFrame({"tf": ["O", "O", "A"], "target": ["K1", "A", "O"], "alpha": [1.0, 0.5, 0.25]})
    return Index(interactions, tf_interactions=tf), tf


def _p_vec(idx, Kt, y, convention):
    P = np.zeros(idx.N)
    for i, p in enumerate(idx.proteins):
        st = idx.offset_y[i]
        total = y[st + 1] + y[st + 2: st + 2 + idx.n_sites[i]].sum()
        P[i] = total
    if convention == "jax":  # driver_map[p2i[name]] = k2i[...] for kinases and orphans
        for k in idx.kinases:
            if k in idx.p2i:
                P[idx.p2i[k]] = Kt[idx.k2i[k]]
        for orphan, proxy in idx.proxy_map.items():
            if orphan in idx.p2i and proxy in idx.k2i:
                P[idx.p2i[orphan]] = Kt[idx.k2i[proxy]]
    else:  # legacy System.rhs: drive by the protein's own name
        for i, p in enumerate(idx.proteins):
            if p in idx.kinases:
                P[i] = Kt[idx.k2i[p]]
            elif p in idx.proxy_map:
                P[i] = Kt[idx.k2i[idx.proxy_map[p]]]
    return P


def test_orphan_driving_divergence_does_not_reach_tf_input():
    from networkmodel.BuildMatrix import build_tf_matrix

    idx, tf_net = _index_with_orphan()
    assert idx.proxy_map == {"O": "K1"}
    orphan_row = idx.proteins.index("O")
    assert idx.p2i["O"] == idx.proteins.index("K1") != orphan_row  # the redirection under test

    rng = np.random.default_rng(0)
    state_dim = int(idx.offset_y[-1] + 2 + idx.n_sites[-1])
    y = rng.uniform(0.1, 2.0, state_dim)
    Kt = rng.uniform(0.5, 3.0, len(idx.kinases))
    p_jax, p_legacy = _p_vec(idx, Kt, y, "jax"), _p_vec(idx, Kt, y, "legacy")
    # The two conventions disagree only at the orphan's own row ...
    differ = np.nonzero(~np.isclose(p_jax, p_legacy))[0].tolist()
    assert differ == [orphan_row]

    # ... and the TF matrix never touches that row or column, so TF inputs (hence every derivative) agree.
    tf_mat = build_tf_matrix(tf_net, idx).toarray()
    assert not tf_mat[:, orphan_row].any() and not tf_mat[orphan_row, :].any()
    np.testing.assert_array_equal(tf_mat @ p_jax, tf_mat @ p_legacy)
