"""N2: bounds come from calculate_bio_bounds only; a [networkmodel.bounds] table is a clear error."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("A_i", "B_i", "C_i", "D_i", "E_i", "c_k", "tf_scale", "Dp_i")


def test_config_bounds_table_rejected(tmp_path):
    from config_loader import load_config_toml

    src = (ROOT / "config.toml").read_text()
    assert "[networkmodel.bounds]" not in src
    p = tmp_path / "c.toml"
    p.write_text(src.replace("[networkmodel.models]", "[networkmodel.bounds]\nc_k = [0.1, 1.0]\n\n[networkmodel.models]", 1))
    with pytest.raises(ValueError, match="N2"):
        load_config_toml(p)


def test_init_raw_params_requires_all_groups():
    from networkmodel.params import init_raw_params

    defaults = {g: np.ones(2) for g in GROUPS}
    defaults["tf_scale"] = np.ones(1)
    bounds = {g: (0.01, 5.0) for g in GROUPS}
    theta0, slices, xl, xu = init_raw_params(defaults, custom_bounds=bounds)
    assert theta0.shape == xl.shape == xu.shape
    del bounds["E_i"]
    with pytest.raises(ValueError, match="N2"):
        init_raw_params(defaults, custom_bounds=bounds)
