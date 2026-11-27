"""Pre-merge item 3: one c_k upper rule for all models, clip(0.1/var(Kmat), 5, 15). A model-specific limit is allowed
only as an entry of networkmodel.utils.CK_MODEL_LIMITS with its documented reason; no other per-model override."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_design_bounds import GRIDS, _inputs

XF = pytest.mark.xfail(strict=True, reason="pre-merge 3: one c_k rule, no silent per-model overrides")


def _ck_hi(monkeypatch, model, var=0.01):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", model)
    idx, df_prot, df_rna, tf_mat, _ = _inputs()
    a = np.sqrt(var)
    kin = SimpleNamespace(Kmat=np.array([[1 + a, 1 - a], [1 - a, 1 + a]]))
    return u.calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin, time_points=GRIDS)["c_k"][1]


@XF
@pytest.mark.parametrize("model", [0, 1, 2, 4])
def test_one_rule_unless_documented(monkeypatch, model):
    import networkmodel.utils as u

    rule = float(np.clip(0.1 / 0.01, 5.0, 15.0))
    limits = u.CK_MODEL_LIMITS
    for m, entry in limits.items():
        assert isinstance(entry["max"], float) and len(entry["reason"]) > 20, (m, entry)
    expected = min(rule, limits[model]["max"]) if model in limits else rule
    assert np.isclose(_ck_hi(monkeypatch, model), expected, rtol=1e-12)
