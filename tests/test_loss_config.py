"""N3: weighted MSE (loss = 0) is the only loss; any other value is a clear error."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _toml_with(tmp_path, old, new):
    src = (ROOT / "config.toml").read_text()
    assert old in src
    p = tmp_path / "c.toml"
    p.write_text(src.replace(old, new, 1))
    return p


def test_shipped_config_uses_weighted_mse():
    from config_loader import load_config_toml

    assert load_config_toml(ROOT / "config.toml").loss_mode == 0


@pytest.mark.parametrize("value", [1, 2, -1, 5])
def test_other_loss_values_are_rejected(tmp_path, value):
    from config_loader import load_config_toml

    with pytest.raises(ValueError, match="N3"):
        load_config_toml(_toml_with(tmp_path, "\nloss = 0\n", f"\nloss = {value}\n"))


def test_loss_function_module_is_gone():
    assert not (ROOT / "networkmodel" / "LossFunction.py").exists()
