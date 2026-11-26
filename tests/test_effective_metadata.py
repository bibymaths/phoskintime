"""META: metadata.json records the effective settings used by the run, not the requested config values."""
import json
from types import SimpleNamespace

import numpy as np


def test_networkmodel_effective_settings_written(tmp_path):
    from common.results import write_metadata
    from networkmodel.runner import _record_effective_settings

    write_metadata(tmp_path, workflow="test", extra={"effective_settings": {"cores": 80, "model": "distributive"}})
    args = SimpleNamespace(output_dir=str(tmp_path), lambda_protein=1.0, lambda_rna=1.0, lambda_phospho=1.0,
                           lambda_prior=0.1, seed=42, pinn_seed=42, model_code=0)
    slices = {"A_i": slice(0, 2), "c_k": slice(2, 3)}
    xl, xu = np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])
    _record_effective_settings(args, {"A_i": (0.1, 2.0), "c_k": (0.01, 5.0)}, slices, xl, xu,
                               {"protein": "uniform", "rna": "uniform"})
    eff = json.loads((tmp_path / "metadata.json").read_text())["effective_settings"]
    assert eff["solver_fit"]["rtol"] == 1e-6 and eff["solver_fit"]["solver_name"] == "Kvaerno4"
    assert eff["solver_export_observation_times"]["rtol"] == 1e-5
    assert eff["solver_steady_state"]["atol"] == 1e-8
    assert eff["bounds"]["A_i"] == {"raw_lower_min": -2.0, "raw_upper_max": 2.0, "physical_lower": 0.1,
                                    "physical_upper": 2.0}
    assert eff["weighting"] == {"protein": "uniform", "rna": "uniform"}
    assert eff["seeds"]["seed"] == 42 and "kinase_drive" in eff and "scaling_method" in eff
    assert "weighted mean squared error" in eff["loss_function"]
    assert eff["cores"] == 80 and eff["model"] == "distributive"  # existing block is merged, not replaced


def test_protwise_effective_settings():
    from protwise.runner.main import _effective_settings

    eff = _effective_settings({"bounds": {"A": [0, 1]}, "seed": 7})
    assert eff["solver"]["rtol"] == 1e-6 and eff["seeds"]["sensitivity"] == 7
    assert eff["rna_times"][0] == 4.0 and "bootstraps_used" not in eff
