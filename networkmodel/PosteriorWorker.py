"""Standalone posterior worker for one NumPyro NUTS chain.

Run as:
    python -m networkmodel.posterior_worker --run-config ... --chain-id 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from networkmodel.PosteriorObjective import build_networkmodel_posterior_context
from networkmodel.BayesianInference import run_numpyro_posterior
from config.config import setup_logger


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--chain-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-warmup", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    args = parser.parse_args()

    with open(args.run_config) as f:
        run_cfg = json.load(f)

    base_output_dir = Path(run_cfg["output_dir"])
    chain_out = base_output_dir / "posterior_chains" / f"chain_{args.chain_id:03d}"
    chain_out.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir=chain_out)

    status_path = chain_out / "chain_status.json"
    status = {
        "chain_id": int(args.chain_id),
        "pid": os.getpid(),
        "seed": int(args.seed),
        "success": False,
        "state": "started",
        "failure_reason": "",
        "posterior_samples": "",
        "posterior_summary": "",
    }
    _write_status(status_path, status)

    try:
        logger.info("[PosteriorWorker] Chain %d started.", args.chain_id)

        status["state"] = "rebuilding_objective"
        _write_status(status_path, status)

        ctx = build_networkmodel_posterior_context(
            run_config_path=args.run_config,
            chain_output_dir=chain_out,
        )

        status["state"] = "running_numpyro"
        _write_status(status_path, status)

        run_numpyro_posterior(
            ctx,
            num_warmup=int(args.num_warmup),
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            num_chains=1,
            chain_method="sequential",
        )

        status.update(
            {
                "success": True,
                "state": "complete",
                "posterior_samples": str(chain_out / "posterior" / "posterior_samples.csv"),
                "posterior_summary": str(chain_out / "posterior" / "posterior_summary.csv"),
            }
        )
        _write_status(status_path, status)

        logger.info("[PosteriorWorker] Chain %d complete.", args.chain_id)

    except Exception as exc:
        logger.exception("[PosteriorWorker] Chain %d failed.", args.chain_id)

        status.update(
            {
                "success": False,
                "state": "failed",
                "failure_reason": repr(exc),
            }
        )
        _write_status(status_path, status)
        raise


if __name__ == "__main__":
    main()
