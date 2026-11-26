"""N16: exported c_k correlation/boxplot and tf_scale distribution use physical parameters
(softplus of the raw optimizer vector), like every other export, not raw optimizer coordinates."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

N16 = pytest.mark.xfail(reason="N16: plots from raw optimizer coordinates", strict=True)

SLICES = {"A_i": slice(0, 2), "B_i": slice(2, 4), "C_i": slice(4, 6), "D_i": slice(6, 8), "E_i": slice(8, 10),
          "c_k": slice(10, 13), "tf_scale": slice(13, 14), "Dp_i": slice(14, 16)}


def _setup(monkeypatch):
    import networkmodel.export as ex

    captured = {"heatmap": [], "boxplot": [], "violinplot": []}
    monkeypatch.setattr(ex.sns, "heatmap", lambda data, **k: captured["heatmap"].append(pd.DataFrame(data).copy()))
    monkeypatch.setattr(ex.sns, "boxplot", lambda data=None, **k: captured["boxplot"].append(pd.DataFrame(data).copy()))
    monkeypatch.setattr(ex.sns, "violinplot",
                        lambda data=None, **k: captured["violinplot"].append(np.asarray(data, float).copy()))
    monkeypatch.setattr(ex.sns, "pairplot", lambda *a, **k: None)
    monkeypatch.setattr(ex.plt, "savefig", lambda *a, **k: None)
    rng = np.random.default_rng(3)
    X = rng.normal(-1.0, 2.0, size=(6, 16))
    idx = SimpleNamespace(kinases=["K0", "K1", "K2"])
    return ex, captured, SimpleNamespace(X=X), idx


def _sp(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


@N16
def test_kinase_correlation_uses_physical_c_k(monkeypatch, tmp_path):
    ex, cap, res, idx = _setup(monkeypatch)
    ex.export_param_correlations(res, SLICES, idx, str(tmp_path))
    expected = pd.DataFrame(_sp(res.X[:, SLICES["c_k"]]), columns=idx.kinases).corr()
    np.testing.assert_allclose(cap["heatmap"][0].to_numpy(), expected.to_numpy(), rtol=1e-12)


@N16
def test_distributions_use_physical_c_k_and_tf_scale(monkeypatch, tmp_path):
    ex, cap, res, idx = _setup(monkeypatch)
    ex.export_parameter_distributions(res, SLICES, idx, str(tmp_path))
    box = cap["boxplot"][0]
    np.testing.assert_allclose(np.sort(box.to_numpy().ravel()), np.sort(_sp(res.X[:, SLICES["c_k"]]).ravel()),
                               rtol=1e-12)
    np.testing.assert_allclose(cap["violinplot"][0].ravel(), _sp(res.X[:, SLICES["tf_scale"]]).ravel(), rtol=1e-12)
