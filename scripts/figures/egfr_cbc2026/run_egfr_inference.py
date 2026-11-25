#!/usr/bin/env python3
"""Run EGFR-specific profiles and full-network NUTS posterior from a saved PhosKinTime fit.

Place this script at the PhosKinTime repository root and run it from the existing
Python environment. It never changes the original fitted model or posterior payload.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunData:
    repo: Path
    source: Path
    payload: Path
    output: Path
    config: dict
    theta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    names: list[str]
    egfr_indices: list[int]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EGFR inference from a saved PhosKinTime networkmodel result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=("check", "profile", "posterior", "both", "summarize"), default="check")
    p.add_argument("--source", default="results_network_distributive", help="existing fitted results directory")
    p.add_argument("--output", default="results_network_distributive/egfr_inference", help="isolated inference output directory")
    p.add_argument("--indices", default="", help="comma-separated EGFR indices to profile; default: all 13")
    p.add_argument("--grid-size", type=positive_int, default=7)
    p.add_argument("--profile-workers", type=positive_int, default=1)
    p.add_argument("--chains", type=positive_int, default=1, help="independent NUTS processes")
    p.add_argument("--warmup", type=positive_int, default=100)
    p.add_argument("--samples", type=positive_int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=0, help="posterior worker timeout seconds; zero disables")
    return p.parse_args()


def resolve_at(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def softplus(raw: np.ndarray | float) -> np.ndarray:
    a = np.asarray(raw, dtype=np.float64)
    return np.logaddexp(0.0, a)


def load_run(repo: Path, args: argparse.Namespace) -> RunData:
    if not (repo / "networkmodel" / "BayesianInference.py").is_file():
        raise RuntimeError("Move this script into the root of the PhosKinTime repository.")
    source = resolve_at(repo, args.source)
    payload = source / "posterior_payload"
    output = resolve_at(repo, args.output)
    if not payload.is_dir():
        raise FileNotFoundError(f"Missing posterior payload directory: {payload}")
    config_file = payload / "posterior_run_config.json"
    config = json.loads(config_file.read_text())
    theta = np.load(payload / "posterior_theta0.npy", allow_pickle=False)
    lower = np.load(payload / "posterior_lower.npy", allow_pickle=False)
    upper = np.load(payload / "posterior_upper.npy", allow_pickle=False)
    names = json.loads((payload / "posterior_parameter_names.json").read_text())
    if theta.ndim != 1 or theta.shape != lower.shape or theta.shape != upper.shape or len(names) != len(theta):
        raise ValueError("Saved theta/bounds/name arrays are incompatible.")
    if not np.isfinite(theta).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Saved theta or bounds contain non-finite values.")
    if not np.all(lower < upper):
        raise ValueError("Saved raw bounds must satisfy lower < upper.")
    if not np.all((lower <= theta) & (theta <= upper)):
        raise ValueError("Saved fitted raw parameters lie outside bounds.")
    egfr = [i for i, n in enumerate(names) if re.search(r"\[EGFR(?:_|\])", n, flags=re.IGNORECASE)]
    if not egfr:
        raise ValueError("No EGFR-labelled parameters found in saved names.")
    # The original payload must remain immutable; derive a separate config for new output.
    if output == source or output == payload or source in output.parents and output.name == "posterior_payload":
        raise ValueError("Choose a separate --output, not the original results or payload directory.")
    return RunData(repo, source, payload, output, config, theta, lower, upper, names, egfr)


def print_preflight(d: RunData) -> None:
    print(f"Repository: {d.repo}")
    print(f"Existing fit: {d.source}")
    print(f"New inference output: {d.output}")
    print(f"Fitted model parameter dimension: {len(d.theta)}")
    print(f"EGFR parameter count: {len(d.egfr_indices)}")
    print(f"Fitted data mode: protein/RNA/phospho (from run config and original model)")
    print("\nEGFR parameters (raw optimization coordinates):")
    for i in d.egfr_indices:
        print(f"  {i:3d}  {d.names[i]:28}  raw={d.theta[i]: 10.6f}  physical={softplus(d.theta[i]).item(): 10.6f}")
    print("\nInput-file checks:")
    for key in ("kinase_net", "tf_net", "ms", "rna", "phospho", "kinopt", "tfopt"):
        value = d.config.get(key)
        if value:
            path = resolve_at(d.repo, value)
            print(f"  {key:11}: {'OK' if path.is_file() else 'MISSING'} {value}")
            if not path.is_file():
                raise FileNotFoundError(f"Inference reconstruction input not found: {path}")
    code = (d.repo / "networkmodel" / "BayesianInference.py").read_text()
    if "def _sanitize_numpyro_posterior_bounds(" in code:
        raise RuntimeError(
            "Local BayesianInference.py still has the obsolete physical-sign bound sanitizer. "
            "Use the corrected raw-theta posterior implementation from commit "
            "6d617a0af080b689f7ee6c86cbb4f7760bfc740c or later. "
            "Do not edit saved parameter bounds to make them positive."
        )
    if "def _theta_to_unit_interval(" not in code or "theta_raw = lower_raw + width_raw * theta_unit" not in code:
        raise RuntimeError(
            "Unrecognized BayesianInference.py posterior API. Check source revision before sampling."
        )
    print("\nSaved raw bounds and corrected posterior parameterization: OK")
    print("No original results have been modified.")


def make_config(d: RunData) -> Path:
    d.output.mkdir(parents=True, exist_ok=True)
    cfg = dict(d.config)
    cfg["output_dir"] = str(d.output)
    cfg["payload_dir"] = str(d.payload)
    path = d.output / "egfr_run_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != cfg:
            raise RuntimeError(f"Existing {path} differs; choose a fresh --output.")
    else:
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    metadata_path = d.output / "egfr_parameter_reference.csv"
    if not metadata_path.exists():
        with metadata_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("index", "parameter", "raw_fitted", "physical_fitted", "raw_lower", "raw_upper"))
            for i in d.egfr_indices:
                writer.writerow((i, d.names[i], d.theta[i], softplus(d.theta[i]).item(), d.lower[i], d.upper[i]))
    return path


def fresh_stage(d: RunData, stage: str) -> None:
    dirs = ("profiles", "profile_workers") if stage == "profile" else ("posterior", "posterior_chains")
    existing = [d.output / name for name in dirs if (d.output / name).exists()]
    if existing:
        raise RuntimeError(
            f"Output for '{stage}' already exists: {existing}. "
            "Choose a new --output path for a clean, independent run. "
            "The script never deletes or overwrites prior inference runs."
        )


def selected_indices(d: RunData, given: str) -> list[int]:
    if not given.strip():
        return d.egfr_indices
    indices = [int(x.strip()) for x in given.split(",") if x.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--indices must be a nonempty comma-separated list without duplicates")
    unexpected = set(indices) - set(d.egfr_indices)
    if unexpected:
        raise ValueError(f"Non-EGFR parameter indices supplied: {sorted(unexpected)}")
    return indices


def plot_profiles(d: RunData) -> None:
    path = d.output / "profiles" / "profile_likelihood_summary.csv"
    if not path.exists():
        return
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(path)
    folder = d.output / "egfr_plots" / "profiles"
    folder.mkdir(parents=True, exist_ok=True)
    for idx in d.egfr_indices:
        name = d.names[idx]
        part = df[df["parameter_name"] == name].copy()
        if part.empty:
            continue
        part = part.sort_values("grid_value")
        x = softplus(part["grid_value"].to_numpy())
        y = part["delta_objective"].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(x, y, marker="o", ms=3, label="profile objective")
        ax.axvline(softplus(d.theta[idx]).item(), ls="--", label="original fitted value")
        ax.set(xlabel=f"{name} (physical parameter scale)", ylabel="Delta weighted objective", title=f"EGFR profile: {name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
        fig.savefig(folder / f"profile_{safe}.png", dpi=180)
        plt.close(fig)
    print(f"EGFR profile plots: {folder}")


def run_profile(d: RunData, cfg_path: Path, a: argparse.Namespace) -> None:
    fresh_stage(d, "profile")
    indices = selected_indices(d, a.indices)
    print(f"\nProfiling {len(indices)} EGFR parameters: {indices}", flush=True)
    from networkmodel.BayesianInference import run_profile_likelihood_standalone_processes
    result = run_profile_likelihood_standalone_processes(
        run_config_path=cfg_path,
        output_dir=d.output,
        parameter_indices=indices,
        grid_size=a.grid_size,
        max_workers=a.profile_workers,
    )
    print("Profile worker outcomes:", flush=True)
    print(result["worker_status"].to_string(index=False), flush=True)
    plot_profiles(d)


def summarize_posterior(d: RunData) -> None:
    import pandas as pd
    path = d.output / "posterior" / "posterior_samples.csv"
    if not path.exists():
        raise FileNotFoundError(f"No posterior samples yet: {path}")
    frame = pd.read_csv(path)
    if "chain" not in frame:
        raise ValueError("Expected a 'chain' column in the combined posterior samples")
    cols = [d.names[i] for i in d.egfr_indices]
    missing = [n for n in cols if n not in frame.columns]
    if missing:
        raise ValueError(f"Posterior samples missing EGFR columns: {missing}")
    selected = ["chain", *cols]
    if "sigma" in frame.columns:
        selected.append("sigma")
    out = d.output / "egfr_posterior_samples.csv"
    frame[selected].to_csv(out, index=False)
    print(f"EGFR posterior samples: {out}")

    diagnostic = {}
    chains = list(frame["chain"].dropna().unique())
    if len(chains) >= 2:
        try:
            from numpyro.diagnostics import summary as numpyro_summary
            count = min((frame["chain"] == ch).sum() for ch in chains)
            if count >= 10:
                chain_samples = {
                    name: np.stack([
                        frame.loc[frame["chain"] == ch, name].to_numpy()[:count]
                        for ch in chains
                    ]) for name in cols
                }
                diagnostic = numpyro_summary(chain_samples, prob=0.90, group_by_chain=True)
        except Exception as exc:
            print(f"WARNING: NumPyro diagnostics unavailable: {exc}")
    else:
        print("WARNING: A single chain cannot establish between-chain R-hat convergence.")
    rows = []
    for i in d.egfr_indices:
        name = d.names[i]
        v = frame[name].dropna().to_numpy(dtype=float)
        row = {
            "parameter": name, "fitted_physical": softplus(d.theta[i]).item(),
            "mean": float(np.mean(v)), "median": float(np.median(v)),
            "sd": float(np.std(v, ddof=1)),
            "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
            "chains": len(chains), "draws_total": len(v),
            "n_eff": float(np.asarray(diagnostic[name]["n_eff"])) if name in diagnostic else np.nan,
            "r_hat": float(np.asarray(diagnostic[name]["r_hat"])) if name in diagnostic else np.nan,
        }
        rows.append(row)
    summary_path = d.output / "egfr_posterior_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"EGFR posterior summary: {summary_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = d.output / "egfr_plots" / "posterior"
    folder.mkdir(parents=True, exist_ok=True)
    for i in d.egfr_indices:
        name = d.names[i]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
        for chain, group in frame.groupby("chain", sort=True):
            series = group[name].to_numpy(dtype=float)
            ax1.plot(np.arange(len(series)), series, alpha=0.65, linewidth=0.8, label=f"chain {chain}")
        ax1.set(title="Trace", xlabel="Posterior draw", ylabel=name)
        if len(chains) < 7:
            ax1.legend(fontsize=7)
        ax2.hist(frame[name].to_numpy(dtype=float), bins=35, density=True, alpha=0.75)
        ax2.axvline(softplus(d.theta[i]).item(), ls="--", label="fitted value")
        ax2.set(title="Marginal posterior", xlabel=f"{name} (physical)", ylabel="Density")
        ax2.legend(fontsize=7)
        fig.suptitle(f"EGFR: {name}")
        fig.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
        fig.savefig(folder / f"posterior_{safe}.png", dpi=180)
        plt.close(fig)
    print(f"EGFR trace and density plots: {folder}")


def run_posterior(d: RunData, cfg_path: Path, a: argparse.Namespace) -> None:
    fresh_stage(d, "posterior")
    print(f"\nSampling FULL {len(d.theta)}-parameter posterior using {a.chains} independent chain(s).", flush=True)
    print("EGFR marginals are extracted after sampling, not sampled in isolation.", flush=True)
    from networkmodel.BayesianInference import run_numpyro_posterior_standalone_processes
    result = run_numpyro_posterior_standalone_processes(
        run_config_path=cfg_path,
        output_dir=d.output,
        num_warmup=a.warmup,
        num_samples=a.samples,
        seed=a.seed,
        num_processes=a.chains,
        timeout_seconds=a.timeout if a.timeout > 0 else None,
    )
    print("Chain outcomes:", flush=True)
    print(result["chain_status"].to_string(index=False), flush=True)
    summarize_posterior(d)


def main() -> int:
    a = parse_args()
    repo = Path(__file__).resolve().parents[3]  # repository root (this file: scripts/figures/egfr_cbc2026/)
    os.chdir(repo)  # relative input paths in the saved run configuration use repository root
    d = load_run(repo, a)
    print_preflight(d)
    if a.stage == "check":
        print("\nPreflight passed. Run --stage profile, posterior, or both.")
        return 0
    if a.stage == "summarize":
        plot_profiles(d)
        summarize_posterior(d)
        return 0
    cfg_path = make_config(d)
    print(f"\nInference configuration: {cfg_path}")
    if a.stage in ("profile", "both"):
        run_profile(d, cfg_path, a)
    if a.stage in ("posterior", "both"):
        run_posterior(d, cfg_path, a)
    print("\nCompleted requested stage(s). Original fitted results remain unchanged.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
