"""N7 (6a): a posterior/profile rebuild reads the run's own effective settings from its metadata.json."""
import json

import pytest


def test_load_run_settings_reads_metadata(tmp_path):
    from networkmodel.run_settings import load_run_settings, solver_config_from

    eff = {"scaling_method": "raw", "time_points": {"protein": [0, 1], "rna": [1], "phospho": [0, 1]},
           "baseline_times": {"protein": 0.0, "rna": 1.0, "phospho": 0.0},
           "solver_fit": {"solver_name": "Kvaerno4", "rtol": 1e-6, "atol": 1e-6, "max_steps": 1000,
                          "root_max_steps": 20}}
    (tmp_path / "metadata.json").write_text(json.dumps({"effective_settings": eff}))
    got = load_run_settings(tmp_path)
    assert got["scaling_method"] == "raw" and solver_config_from(got).rtol == 1e-6


def test_runs_without_recorded_settings_are_refused(tmp_path):
    from networkmodel.run_settings import load_run_settings

    (tmp_path / "metadata.json").write_text(json.dumps({"effective_settings": {"scaling_method": "fc_start"}}))
    with pytest.raises(ValueError, match="N7"):
        load_run_settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_run_settings(tmp_path / "missing")


def test_rebuild_uses_run_settings_not_config():
    import inspect

    import networkmodel.PosteriorObjective as po

    src = inspect.getsource(po.build_networkmodel_posterior_context)
    assert "load_run_settings(base_output_dir)" in src and "solver_config=solver_config_from(run_eff)" in src
    assert "TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO" not in src
