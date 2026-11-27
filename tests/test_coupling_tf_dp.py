"""Pre-merge item 2: A_i bounds include the TF modulation factor, C_i bounds the phospho outflux.

TF modulation (backend._synth): u = u_raw/(1+|u_raw|) in (-1, 1); activation f = 1 + tf u/(1+u) in [1, 1 + tf/2),
repression f = 1/(1 + tf |u|) in (1/(1 + tf), 1]. Steady state R = A f / B, so with tf_max = the tf_scale upper bound
(after model limits) and any TF edge in the network:
  A_i = [B_min R_lo / (1 + tf_max/2), B_max R_hi (1 + tf_max)]          (no TF edge: f = 1, as before)
Observed protein = total T over all states. dT/dt = trans - D T - sum_j Dp_j x_j (E moves sites back to the
unphosphorylated pool inside T: no net term), so T = trans / (D + Dp_eff), 0 <= Dp_eff <= kappa Dp_max with
kappa = 1 (models 0, 1, 4: one Dp per state, states sum to T) or kappa = max sites per protein (model 2: a state decays
D + sum of its sites' Dp):
  C_i = [D_min P_lo / R_hi, (D_max + kappa Dp_max) P_hi / R_lo]          (model 4: x (1 + R) as before)
"""

import numpy as np
import pytest
import scipy.sparse as sp

from tests.test_design_bounds import GRIDS, _inputs

S = 1.5


def _net(with_tf=True):
    idx, df_prot, df_rna, _, kin = _inputs()
    idx.n_sites = [3, 1]
    tf = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]) if with_tf else np.zeros((2, 2)))
    return idx, df_prot, df_rna, tf, kin


@pytest.mark.parametrize("model", [0, 1, 2, 4])
def test_a_and_c_bounds(monkeypatch, model):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", model)
    idx, df_prot, df_rna, tf, kin = _net()
    b = u.calculate_bio_bounds(idx, df_prot, df_rna, tf, kin, time_points=GRIDS)
    r_lo, r_hi = min(1.0, df_rna.fc.min()) / S, max(1.0, df_rna.fc.max()) * S
    p_lo, p_hi = min(1.0, df_prot.fc.min()) / S, max(1.0, df_prot.fc.max()) * S
    (b_lo, b_hi), (d_lo, d_hi), tf_hi, dp_hi = b["B_i"], b["D_i"], b["tf_scale"][1], b["Dp_i"][1]
    np.testing.assert_allclose(b["A_i"], (b_lo * r_lo / (1 + tf_hi / 2), b_hi * r_hi * (1 + tf_hi)), rtol=1e-12)
    kappa = 3 if model == 2 else 1
    c = (d_lo * p_lo / r_hi, (d_hi + kappa * dp_hi) * p_hi / r_lo)
    if model == 4:
        c = (c[0] * (1 + r_hi), c[1] * (1 + r_lo))
    np.testing.assert_allclose(b["C_i"], c, rtol=1e-12)


def test_no_tf_edge_keeps_factor_one(monkeypatch):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", 0)
    idx, df_prot, df_rna, tf, kin = _net(with_tf=False)
    b = u.calculate_bio_bounds(idx, df_prot, df_rna, tf, kin, time_points=GRIDS)
    r_lo, r_hi = min(1.0, df_rna.fc.min()) / S, max(1.0, df_rna.fc.max()) * S
    np.testing.assert_allclose(b["A_i"], (b["B_i"][0] * r_lo, b["B_i"][1] * r_hi), rtol=1e-12)
    np.testing.assert_allclose(b["C_i"][1], (b["D_i"][1] + b["Dp_i"][1]) * (max(1.0, df_prot.fc.max()) * S)
                               / (min(1.0, df_rna.fc.min()) / S), rtol=1e-12)
