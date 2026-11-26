"""N4: uniform weighting is the only weighting; any other setting is a clear error."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_uniform_accepted():
    from networkmodel.OptimizationProblem import build_weight_functions

    assert build_weight_functions("uniform", "uniform")["protein"] == "uniform"


@pytest.mark.parametrize("prot,rna", [("inverse", "uniform"), ("uniform", "early_emphasis")])
def test_other_methods_rejected(prot, rna):
    from networkmodel.OptimizationProblem import build_weight_functions

    with pytest.raises(ValueError, match="N4"):
        build_weight_functions(prot, rna)


def test_config_rejects_non_uniform(tmp_path):
    from config_loader import load_config_toml

    src = (ROOT / "config.toml").read_text()
    p = tmp_path / "c.toml"
    p.write_text(src.replace('weighting_method_rna     = "uniform"', 'weighting_method_rna = "inverse"', 1))
    with pytest.raises(ValueError, match="N4"):
        load_config_toml(p)
