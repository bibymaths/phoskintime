"""N7 (option A): data and model are both normalised to their own value at the baseline time.
Baseline times are defined once (t = 0 for protein/phospho, the first RNA time for RNA); scaling_method = fc_start;
zero, near-zero or NaN baselines are skipped with a count, never divided silently."""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_config_uses_fc_start():
    from config_loader import load_config_toml

    assert load_config_toml(ROOT / "config.toml").scaling_method == "fc_start"


def test_baseline_times_defined_once():
    import networkmodel.config as nc
    from networkmodel.baselines import BASELINE_TIMES, baseline_index

    assert BASELINE_TIMES == {"protein": float(nc.TIME_POINTS_PROTEIN[0]), "phospho": float(nc.TIME_POINTS_PHOSPHO[0]),
                              "rna": float(nc.TIME_POINTS_RNA[0])}
    times = np.array([0.0, 0.5, 4.0, 8.0])
    assert baseline_index(times, "rna") == 2 and baseline_index(times, "protein") == 0


@pytest.mark.parametrize("module", ["networkmodel.runner", "networkmodel.PosteriorObjective", "networkmodel.simulate"])
def test_no_hard_coded_baseline_times(module):
    import importlib

    src = inspect.getsource(importlib.import_module(module))
    assert "_base_idx(solver_times, 4.0)" not in src and "_bidx(4.0)" not in src


def test_fc_start_skips_bad_baselines_and_counts_them():
    from networkmodel.utils import process_and_scale_raw_data

    df = pd.DataFrame({"GeneID": ["A", "B", "C", "D"], "Psite": ["", "", "", ""],
                       "x1": [2.0, 0.0, np.nan, 1e-15], "x2": [4.0, 1.0, 1.0, 1.0], "x3": [1.0, 1.0, 1.0, 1.0]})
    report = {}
    out = process_and_scale_raw_data(df, [0.0, 1.0, 2.0], ["GeneID", "Psite"], scale_method="fc_start",
                                     report=report)
    assert set(out["GeneID"]) == {"A"}
    np.testing.assert_allclose(out.sort_values("time")["fc"].to_numpy(), [1.0, 2.0, 0.5])
    assert report["skipped_series"] == 3 and report["baseline_time"] == 0.0
