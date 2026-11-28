"""DECISIONS_C1_REFINE A: iterative reference refinement. If a profile point beats the data-only reference by more
than 1e-5, the reference is refitted from the best such point and all profiles are recomputed; at most 3 rounds;
parameters still beaten afterwards are "not assessed (reference unstable)"."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import networkmodel.BayesianInference as bi


def test_points_beating_reference_uses_the_noise_tolerance():
    from networkmodel.utils import REFERENCE_BEAT_TOL, points_beating_reference

    assert REFERENCE_BEAT_TOL == 1e-5
    s = pd.DataFrame({"parameter_name": ["a", "a", "b", "c"],
                      "objective_value": [0.99, 1.0 - 5e-6, 0.98, np.inf]})
    b = points_beating_reference(s, f_ref=1.0)
    assert list(b["parameter_name"]) == ["b", "a"]  # sorted by objective; 1 - 5e-6 is within the noise


def _fake(tmp_path, beat_rounds, beat_param=0):
    """Fake workers: the reference worker returns f_ref = 1.0, 0.9, 0.8, ... per call; parameter workers write one
    point at f_ref + 0.01, and parameter ``beat_param`` also one point at f_ref - 0.05 while the round index is in
    ``beat_rounds``."""
    calls = {"ref": 0, "starts": []}

    class P:
        def __init__(self, cmd, stdout=None, stderr=None, cwd=None, env=None):
            self.cmd, self.returncode, self.pid = cmd, None, 1
            self.idx = int(cmd[cmd.index("--parameter-index") + 1]) if "--parameter-index" in cmd else -1

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            ref_path = tmp_path / "profiles" / "profile_reference.json"
            if self.idx < 0:
                if "--starts-file" in self.cmd:
                    calls["starts"].append(np.load(self.cmd[self.cmd.index("--starts-file") + 1]).tolist())
                f = 1.0 - 0.1 * calls["ref"]
                ref_path.write_text(json.dumps({"f_ref": f, "theta_ref": [0.0, 0.0], "status": "converged",
                                                "threshold": 0.1}))
                calls["ref"] += 1
            else:
                f_ref = json.loads(ref_path.read_text())["f_ref"]
                rnd = calls["ref"] - 1
                rows = [{"parameter_name": f"param_{self.idx}", "parameter_index": self.idx, "side": "above",
                         "grid_value_physical": 2.0, "objective_value": f_ref + 0.01, "success": True,
                         "convergence_status": "converged", "optimized_free_parameters": json.dumps([1.0, 1.0])}]
                if self.idx == beat_param and rnd in beat_rounds:
                    rows.append({**rows[0], "objective_value": f_ref - 0.05,
                                 "optimized_free_parameters": json.dumps([7.0, float(rnd)])})
                d = tmp_path / "profile_workers" / f"param_{self.idx:04d}"
                d.mkdir(parents=True, exist_ok=True)
                csv = d / f"profile_likelihood_param_{self.idx}.csv"
                pd.DataFrame(rows).to_csv(csv, index=False)
                (d / "profile_status.json").write_text(json.dumps({"parameter_index": self.idx, "success": True,
                                                                   "state": "complete", "result_csv": str(csv)}))
            self.returncode = 0
            return 0

    return P, calls


def _run(tmp_path, monkeypatch, beat_rounds):
    rc = tmp_path / "posterior_payload" / "posterior_run_config.json"
    rc.parent.mkdir()
    rc.write_text("{}")
    P, calls = _fake(tmp_path, beat_rounds)
    monkeypatch.setattr(bi.subprocess, "Popen", P)
    monkeypatch.setattr(bi.time, "sleep", lambda s: None)
    out = bi.run_profile_likelihood_standalone_processes(run_config_path=rc, output_dir=tmp_path,
                                                         parameter_indices=[0, 1], grid_size=2, max_workers=2)
    return out, calls


def test_one_refinement_round_when_the_new_reference_holds(tmp_path, monkeypatch):
    out, calls = _run(tmp_path, monkeypatch, beat_rounds={0})
    assert out["refinement"]["rounds"] == 1 and out["refinement"]["unstable_parameters"] == []
    assert calls["ref"] == 2
    assert calls["starts"][0][0] == [7.0, 0.0]  # refit from the best beating point (then the previous reference)
    assert out["reference"]["f_ref"] == pytest.approx(0.9)
    assert (out["summary"]["objective_value"] >= 0.9).all()  # profiles recomputed against the new reference


def test_at_most_three_rounds_then_unstable(tmp_path, monkeypatch):
    out, calls = _run(tmp_path, monkeypatch, beat_rounds={0, 1, 2, 3})
    assert out["refinement"]["rounds"] == 3 and calls["ref"] == 4
    assert out["refinement"]["unstable_parameters"] == ["param_0"]
    assert len(out["refinement"]["history"]) == 4


def test_no_round_when_nothing_beats_the_reference(tmp_path, monkeypatch):
    out, calls = _run(tmp_path, monkeypatch, beat_rounds=set())
    assert out["refinement"]["rounds"] == 0 and calls["ref"] == 1


def test_unstable_label_is_not_an_estimate():
    from networkmodel.utils import UNSTABLE_REFERENCE_LABEL

    assert UNSTABLE_REFERENCE_LABEL == "not assessed (reference unstable)"
