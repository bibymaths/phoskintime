from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import networkmodel.BayesianInference as bi
from networkmodel.BayesianInference import InferenceContext
from networkmodel.PosteriorObjective import write_posterior_payload
from networkmodel.backend import DataMode
from networkmodel.sensitivity import compute_bounds


def _mode() -> DataMode:
    return DataMode(("protein",), "protein", False, True, False)


def _ctx(tmp_path) -> InferenceContext:
    return InferenceContext(
        objective_fun=lambda x: jnp.sum(jnp.asarray(x) ** 2),
        theta0=np.asarray([-0.5, 0.0, 2.0]),
        lower=np.asarray([-1.0, -0.25, 1.0]),
        upper=np.asarray([1.0, 0.25, 3.0]),
        mode=_mode(),
        output_dir=tmp_path,
        parameter_names=("c_k", "A_i", "tf_scale"),
        maxiter=2,
        tol=1e-5,
    )


def test_profile_likelihood_standalone_processes_merges_worker_outputs(tmp_path, monkeypatch):
    run_config = tmp_path / "posterior_payload" / "posterior_run_config.json"
    run_config.parent.mkdir()
    run_config.write_text("{}")

    class FakeProfileProcess:
        def __init__(self, cmd, stdout=None, stderr=None, cwd=None, env=None):
            self.cmd = cmd
            self.returncode = None
            self.pid = 1234
            self.param_idx = int(cmd[cmd.index("--parameter-index") + 1])

        def poll(self):
            if self.returncode is None:
                worker_dir = tmp_path / "profile_workers" / f"param_{self.param_idx:04d}"
                csv_path = worker_dir / f"profile_likelihood_param_{self.param_idx}.csv"
                pd.DataFrame(
                    {
                        "parameter_name": [f"param_{self.param_idx}"],
                        "parameter_index": [self.param_idx],
                        "grid_value": [0.25],
                        "objective_value": [1.5 + self.param_idx],
                        "success": [True],
                    }
                ).to_csv(csv_path, index=False)
                (worker_dir / "profile_status.json").write_text(
                    json.dumps(
                        {
                            "parameter_index": self.param_idx,
                            "success": True,
                            "state": "done",
                            "failure_reason": "",
                            "result_csv": str(csv_path),
                        }
                    )
                )
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(bi.subprocess, "Popen", FakeProfileProcess)
    result = bi.run_profile_likelihood_standalone_processes(
        run_config_path=run_config,
        output_dir=tmp_path,
        parameter_indices=[0, 2],
        grid_size=2,
        max_workers=2,
    )

    assert (tmp_path / "profiles" / "profile_likelihood_summary.csv").exists()
    assert (tmp_path / "profiles" / "profile_worker_status.csv").exists()
    assert set(result["summary"]["parameter_index"]) == {0, 2}
    assert result["summary"].loc[result["summary"]["parameter_index"] == 2, "objective_value"].iloc[0] == pytest.approx(3.5)


def test_numpyro_posterior_standalone_processes_merges_chains_and_all_failures_raise(tmp_path, monkeypatch):
    run_config = tmp_path / "posterior_payload" / "posterior_run_config.json"
    run_config.parent.mkdir()
    run_config.write_text("{}")

    class FakePosteriorProcess:
        fail_all = False

        def __init__(self, cmd, stdout=None, stderr=None, cwd=None, env=None):
            self.cmd = cmd
            self.returncode = None
            self.pid = 2345
            self.chain_id = int(cmd[cmd.index("--chain-id") + 1])

        def wait(self, timeout=None):
            chain_dir = tmp_path / "posterior_chains" / f"chain_{self.chain_id:03d}"
            samples = chain_dir / "posterior" / "posterior_samples.csv"
            if self.fail_all:
                status = {
                    "chain_id": self.chain_id,
                    "seed": 10 + self.chain_id,
                    "success": False,
                    "failure_reason": "synthetic failure",
                    "posterior_samples": "",
                    "posterior_summary": "",
                }
                self.returncode = 1
            else:
                samples.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"theta0": [0.1 + self.chain_id], "sigma": [0.2], "scalar_objective": [1.0], "data_mode": ["protein"]}
                ).to_csv(samples, index=False)
                status = {
                    "chain_id": self.chain_id,
                    "seed": 10 + self.chain_id,
                    "success": True,
                    "failure_reason": "",
                    "posterior_samples": str(samples),
                    "posterior_summary": "",
                }
                self.returncode = 0
            (chain_dir / "chain_status.json").write_text(json.dumps(status))
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(bi.subprocess, "Popen", FakePosteriorProcess)
    result = bi.run_numpyro_posterior_standalone_processes(
        run_config_path=run_config,
        output_dir=tmp_path,
        num_processes=2,
        seed=10,
    )
    assert (tmp_path / "posterior" / "posterior_samples.csv").exists()
    assert (tmp_path / "posterior" / "posterior_chain_status.csv").exists()
    assert set(result["samples"]["chain"]) == {0, 1}

    fail_dir = tmp_path / "all_fail"
    fail_config = fail_dir / "posterior_payload" / "posterior_run_config.json"
    fail_config.parent.mkdir(parents=True)
    fail_config.write_text("{}")
    FakePosteriorProcess.fail_all = True
    with pytest.raises(RuntimeError, match="All standalone posterior chains failed"):
        bi.run_numpyro_posterior_standalone_processes(run_config_path=fail_config, output_dir=fail_dir, num_processes=1)


def test_write_posterior_payload_writes_arrays_names_and_run_config(tmp_path):
    ctx = _ctx(tmp_path)
    args = SimpleNamespace(
        kinase_net="kin.tsv",
        tf_net="tf.tsv",
        ms="ms.tsv",
        rna="rna.tsv",
        phospho="phospho.tsv",
        kinopt="kinopt.csv",
        tfopt="tfopt.csv",
        cores=3,
        n_gen=7,
        seed=11,
        normalize_fc_steady=True,
        use_initial_condition_from_data=False,
    )
    path = write_posterior_payload(
        ctx=ctx,
        runner_args=args,
        lambdas={"protein": 1, "rna": 2, "phospho": 3, "prior": 4},
        output_dir=tmp_path,
    )
    payload_dir = tmp_path / "posterior_payload"
    assert (payload_dir / "posterior_theta0.npy").exists()
    assert (payload_dir / "posterior_lower.npy").exists()
    assert (payload_dir / "posterior_upper.npy").exists()
    assert (payload_dir / "posterior_parameter_names.json").exists()
    data = json.loads(path.read_text())
    assert data["output_dir"] == str(tmp_path)
    assert data["payload_dir"] == str(payload_dir)
    assert data["lambdas"] == {"protein": 1.0, "rna": 2.0, "phospho": 3.0, "prior": 4.0}


def test_param_names_pad_truncate_and_generate_defaults():
    assert bi._param_names(2, None) == ["param_0", "param_1"]
    assert bi._param_names(3, ["a"]) == ["a", "param_1", "param_2"]
    assert bi._param_names(2, ["a", "b", "c"]) == ["a", "b"]


def test_posterior_profile_bounds_preserve_raw_theta_bounds_and_repair_invalid_intervals():
    lower, upper = bi._posterior_profile_bounds(
        np.asarray([-2.0, -1.0, 5.0]),
        np.asarray([-1.0, -2.0, 5.0]),
        ["A_i_0", "tf_scale", "c_k_0"],
    )
    assert lower.tolist() == pytest.approx([-2.0, -1.0, 5.0])
    assert upper[0] == pytest.approx(-1.0)
    assert upper[1] > lower[1]
    assert upper[2] > lower[2]


def test_posterior_plotting_writes_credible_intervals_and_caps_individual_plots(tmp_path):
    samples = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [1.0, 1.1, 1.2], "sigma": [0.5, 0.6, 0.7]})
    summary = pd.DataFrame(
        {
            "parameter": ["a", "b", "sigma"],
            "median": [0.2, 1.1, 0.6],
            "ci_05": [0.11, 1.01, 0.51],
            "ci_95": [0.29, 1.19, 0.69],
        }
    )
    bi._plot_posterior(samples, summary, tmp_path)
    assert (tmp_path / "credible_intervals.png").exists()
    assert (tmp_path / "density" / "trace_a.png").exists()


def test_sensitivity_compute_bounds_signed_and_nonnegative_parameters():
    problem = compute_bounds(
        {
            "c_k": np.asarray([-2.0, 0.0]),
            "A_i": np.asarray([2.0, 0.0]),
            "B_i": np.asarray([3.0]),
            "C_i": np.asarray([4.0]),
            "D_i": np.asarray([5.0]),
            "Dp_i": np.asarray([6.0]),
            "E_i": np.asarray([7.0]),
            "tf_scale": 0.0,
        },
        perturbation=0.5,
    )
    bounds = dict(zip(problem["names"], problem["bounds"]))
    assert bounds["c_k_0"][0] < 0.0
    assert bounds["c_k_1"] == pytest.approx([-0.01, 0.01])
    for name in ["A_i_0", "A_i_1", "B_i_0", "C_i_0", "D_i_0", "Dp_i_0", "E_i_0", "tf_scale"]:
        assert bounds[name][0] >= 0.0
    assert bounds["A_i_1"] == pytest.approx([0.0, 0.01])
    assert all(ub > lb for lb, ub in problem["bounds"])


def test_inference_modules_import_without_running_heavy_workflows():
    import networkmodel.BayesianInference  # noqa: F401
    import networkmodel.PosteriorObjective  # noqa: F401
    import networkmodel.PosteriorWorker  # noqa: F401
    import networkmodel.ProfileWorker  # noqa: F401
    import protwise.paramest.BayesianInference  # noqa: F401
