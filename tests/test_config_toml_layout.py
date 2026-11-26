"""Structural checks on the shipped config.toml (X8: inference keys under [networkmodel]; misplaced keys warn)."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

CONFIG_TOML = Path(__file__).resolve().parents[1] / "config.toml"

# Keys that config_loader.load_config_toml reads from the [networkmodel] table.
NETWORKMODEL_INFERENCE_KEYS = (
    "n_starts",
    "profile_likelihood",
    "profile_indices",
    "profile_grid_size",
    "posterior_sampling",
    "posterior_num_warmup",
    "posterior_num_samples",
)


def test_networkmodel_inference_keys_are_not_under_hyperedge_preprocessing():
    """None of the networkmodel inference keys may sit under [networkmodel.hyperedge_preprocessing]."""
    with CONFIG_TOML.open("rb") as fh:
        cfg = tomllib.load(fh)
    hyperedge = cfg.get("networkmodel", {}).get("hyperedge_preprocessing", {})

    misplaced = [key for key in NETWORKMODEL_INFERENCE_KEYS if key in hyperedge]

    assert misplaced == []


def test_misplaced_known_key_warns(tmp_path):
    from config_loader import load_config_toml

    src = CONFIG_TOML.read_text()
    p = tmp_path / "c.toml"
    p.write_text(src.replace("[networkmodel.hyperedge_preprocessing]\n",
                             "[networkmodel.hyperedge_preprocessing]\nprofile_grid_size = 3\n", 1))
    with pytest.warns(UserWarning, match="X8"):
        load_config_toml(p)


def test_shipped_config_loads_without_misplaced_key_warning(recwarn):
    from config_loader import load_config_toml

    cfg = load_config_toml(CONFIG_TOML)
    assert cfg.profile_grid_size == 10
    assert not [w for w in recwarn if "X8" in str(w.message)]
