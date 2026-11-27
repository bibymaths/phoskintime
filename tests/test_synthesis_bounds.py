"""N22 (P3 decision 2): A_i and C_i bounds are coupled to the rate bounds and the data range.

Steady state without TF modulation: R = A / B and P = C R / D (saturating model: P = C R / ((1 + R) D)).
With the data range widened by s = 1.5, R in [R_lo, R_hi] = [min(1, min fc_rna) / s, max(1, max fc_rna) * s]
(P likewise from fc_prot), the box must contain every A and C consistent with such a steady state for every B, D in
their (design) boxes:
  A_i = [B_min R_lo, B_max R_hi]
  C_i = [D_min P_lo / R_hi, D_max P_hi / R_lo]                          (models 0, 1, 2)
  C_i = [D_min P_lo (1 + R_hi) / R_hi, D_max P_hi (1 + R_lo) / R_lo]    (model 4)
"""
import numpy as np
import pytest

from tests.test_design_bounds import GRIDS, _inputs

pytestmark = pytest.mark.xfail(strict=True, reason="N22-D2: A_i/C_i coupled to rate bounds and data range")

S = 1.5


@pytest.mark.parametrize("model", [0, 1, 2, 4])
def test_synthesis_bounds_follow_rates_and_data(monkeypatch, model):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", model)
    idx, df_prot, df_rna, tf_mat, kin = _inputs()
    b = u.calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin, time_points=GRIDS)
    r_lo, r_hi = min(1.0, df_rna.fc.min()) / S, max(1.0, df_rna.fc.max()) * S
    p_lo, p_hi = min(1.0, df_prot.fc.min()) / S, max(1.0, df_prot.fc.max()) * S
    (b_lo, b_hi), (d_lo, d_hi) = b["B_i"], b["D_i"]
    np.testing.assert_allclose(b["A_i"], (b_lo * r_lo, b_hi * r_hi), rtol=1e-12)
    if model == 4:
        c = (d_lo * p_lo * (1 + r_hi) / r_hi, d_hi * p_hi * (1 + r_lo) / r_lo)
    else:
        c = (d_lo * p_lo / r_hi, d_hi * p_hi / r_lo)
    np.testing.assert_allclose(b["C_i"], c, rtol=1e-12)


def test_steady_state_of_every_rate_is_inside_the_box():
    import networkmodel.utils as u

    idx, df_prot, df_rna, tf_mat, kin = _inputs()
    b = u.calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin, time_points=GRIDS)
    rng = np.random.default_rng(0)
    for _ in range(200):
        B = np.exp(rng.uniform(*np.log(b["B_i"])))
        D = np.exp(rng.uniform(*np.log(b["D_i"])))
        R = rng.uniform(df_rna.fc.min(), df_rna.fc.max())
        P = rng.uniform(df_prot.fc.min(), df_prot.fc.max())
        assert b["A_i"][0] <= B * R <= b["A_i"][1]
        assert b["C_i"][0] <= D * P / R <= b["C_i"][1]
