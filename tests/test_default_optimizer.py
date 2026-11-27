"""N22 / O01 (P3 decision 6): L-BFGS-B with >= 5 seeded starts is the default (best start kept); every start gets a
status: converged (KKT residual <= pgtol), near_converged (pgtol < KKT <= 1e-5, finite run), not_converged.
ProjectedGradient stays selectable; n_gen stays 200."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]



def test_config_defaults():
    from config_loader import load_config_toml

    cfg = load_config_toml(ROOT / "config.toml")
    assert cfg.optimizer == "lbfgsb"
    assert cfg.n_starts >= 5
    assert cfg.maximum_iterations == 200  # n_gen not raised



def test_loader_fallback_defaults(tmp_path):
    from config_loader import load_config_toml

    src = (ROOT / "config.toml").read_text()
    lines = [ln for ln in src.splitlines() if not ln.strip().startswith(("optimizer =", "n_starts ="))]
    p = tmp_path / "c.toml"
    p.write_text("\n".join(lines) + "\n")
    cfg = load_config_toml(p)
    assert cfg.optimizer == "lbfgsb" and cfg.n_starts == 5



@pytest.mark.parametrize("status,kkt,reason,expected", [
    ("converged", 5e-7, "", "converged"),
    ("not_converged", 3e-6, "ABNORMAL", "near_converged"),
    ("not_converged", 1e-5, "", "near_converged"),
    ("not_converged", 2e-5, "", "not_converged"),
    ("not_converged", 3e-6, "x; 2 non-finite gradient evaluation(s)", "not_converged"),
    ("not_converged", float("nan"), "", "not_converged"),
])
def test_convergence_class(status, kkt, reason, expected):
    from networkmodel.optimizers import NEAR_CONVERGED_KKT, convergence_class

    assert NEAR_CONVERGED_KKT == 1e-5
    assert convergence_class(status, kkt, reason) == expected


def test_projected_gradient_still_selectable():
    from networkmodel.optimizers import minimize_bounded

    res = minimize_bounded(lambda x: ((x - 0.3) ** 2).sum(), np.zeros(2), -np.ones(2), np.ones(2),
                           optimizer="projected_gradient", maxiter=500, tol=1e-10)
    np.testing.assert_allclose(res.theta, 0.3, atol=1e-6)
