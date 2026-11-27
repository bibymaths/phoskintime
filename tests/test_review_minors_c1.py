"""Pre-merge review minors: m1 (kappa silently 1 for model 2 without n_sites) and m6 (kinase table without an
is-estimate column)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_design_bounds import GRIDS, _inputs


def test_model2_bounds_need_n_sites(monkeypatch):
    import networkmodel.utils as u

    monkeypatch.setattr(u, "MODEL", 2)
    idx, df_prot, df_rna, tf_mat, kin = _inputs()
    idx = SimpleNamespace(N=idx.N)  # no n_sites
    with pytest.raises(ValueError, match="n_sites"):
        u.calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin, time_points=GRIDS)


def test_kinase_table_has_is_estimate(tmp_path):
    from networkmodel.export import annotate_parameter_tables
    from tests.test_parameter_status import _case
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"],
                          identifiability={"c_k[K1]": "estimate (two-sided)"})
    pd.DataFrame({"Kinase": ["K1", "K2", "K3"], "Activity_Scale_ck": [0.3, 1e-4, 0.016]}).to_csv(
        tmp_path / "model_parameters_kinases.csv", index=False)
    annotate_parameter_tables(str(tmp_path), st)
    k = pd.read_csv(tmp_path / "model_parameters_kinases.csv").set_index("Kinase")
    assert k["Is_estimate"].tolist() == [True, False, False]
