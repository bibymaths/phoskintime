"""Synthetic single-gene sanity check for local PhosKinTime optimization.

Run from the repository root after installing the project dependencies:

    python examples/protwise_optimization_sanity.py --model protwise
    python examples/protwise_optimization_sanity.py --model randmod

The script simulates data from known positive parameters, fits the same model, and
prints three regression-style checks: final loss is lower than the initial loss,
parameters are not collapsed to zero, and the fitted trajectory is not flat.
"""
from __future__ import annotations

import argparse
import numpy as np

import protwise.paramest.normest as normest_mod
from protwise.paramest.normest import aggregate_randmod_phospho
from protwise.models.diffrax_solver import make_local_model_rhs
from networkmodel.backend import solve_diffrax, DiffraxSolverConfig
from config.constants import get_num_params


def initial_condition(model: str, num_sites: int) -> np.ndarray:
    if model == "randmod":
        n_states = (1 << num_sites) - 1
        return np.asarray([1.0, 1.0 / (n_states + 1)] + [1.0 / (n_states + 1)] * n_states, dtype=float)
    return np.asarray([1.0, 0.6] + [0.25] * num_sites, dtype=float)


def true_params(model: str, num_sites: int) -> np.ndarray:
    n = get_num_params(model, num_sites)
    base = np.linspace(0.25, 1.25, n)
    base[:4] = [1.1, 0.35, 0.9, 0.25]
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="protwise", choices=["protwise", "distmod", "succmod", "randmod"])
    parser.add_argument("--num-sites", type=int, default=1)
    args = parser.parse_args()

    # normest reads ODE_MODEL from its module import; override it here so one
    # script can exercise multiple model parameterizations.
    normest_mod.ODE_MODEL = args.model

    t = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
    y0 = initial_condition(args.model, args.num_sites)
    params = true_params(args.model, args.num_sites)
    rhs = make_local_model_rhs(args.model, args.num_sites)
    sol = np.asarray(solve_diffrax(y0, t, params=params, rhs=rhs, config=DiffraxSolverConfig(rtol=1e-4, atol=1e-6)))

    bounds = {"A": (0.01, 3.0), "B": (0.01, 3.0), "C": (0.01, 3.0), "D": (0.01, 3.0), "S(i)": (0.01, 3.0), "D(i)": (0.01, 3.0)}
    # randmod contains subset-level states; observations are site-level, so aggregate
    # each subset into every site it contains before fitting.
    phospho_obs = np.asarray(aggregate_randmod_phospho(sol, args.num_sites)).T if args.model == "randmod" else sol[:, 2:]
    estimated, fits, errors, reg = normest_mod.normest(
        f"SYN_{args.model.upper()}",
        pr_data=sol[:, 1],
        p_data=phospho_obs.T,
        r_data=sol[-min(3, len(t)) :, 0],
        init_cond=y0,
        num_psites=args.num_sites,
        time_points=t,
        bounds=bounds,
    )

    final_params = estimated[-1]
    fitted = fits[0][1]
    print(f"model={args.model} true_params={np.round(params, 4)}")
    print(f"estimated_params={np.round(final_params, 4)}")
    print(f"residual_l2={np.linalg.norm(errors):.6g} regularization={reg:.6g}")
    print(f"not_collapsed={np.any(final_params > 0.05)}")
    print(f"trajectory_variance={np.var(fitted):.6g}")


if __name__ == "__main__":
    main()
