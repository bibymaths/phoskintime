"""Structural checks on the shipped config.toml (TOML parsing only; no project modules are imported)."""
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "X8: the networkmodel inference keys sit under [networkmodel.hyperedge_preprocessing] "
        "in config.toml, so the loader (which reads [networkmodel]) ignores them and uses its defaults"
    ),
)
def test_networkmodel_inference_keys_are_not_under_hyperedge_preprocessing():
    """None of the networkmodel inference keys may sit under [networkmodel.hyperedge_preprocessing]."""
    with CONFIG_TOML.open("rb") as fh:
        cfg = tomllib.load(fh)
    hyperedge = cfg.get("networkmodel", {}).get("hyperedge_preprocessing", {})

    misplaced = [key for key in NETWORKMODEL_INFERENCE_KEYS if key in hyperedge]

    assert misplaced == []
