import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed  # changed: ThreadPoolExecutor
import numpy as np
from SALib.sample import morris
from SALib.analyze.morris import analyze
from numba import njit

from config.constants import (
    ODE_MODEL,
    NUM_TRAJECTORIES,
    PARAMETER_SPACE,
    TIME_POINTS_RNA,
    PERTURBATIONS_VALUE,
    OUT_DIR,
    Y_METRIC,
    get_num_params,
    get_param_names,
)
from protwise.models import solve_ode
from protwise.plotting import Plotter
from config.logconf import setup_logger

logger = setup_logger()


def compute_bound(value, perturbation: float = PERTURBATIONS_VALUE):
    """
    Compute lower and upper bounds for a parameter value for sensitivity analysis.

    For near-zero values, uses a small positive fallback interval [0, 0.1].
    """
    if abs(value) < 1e-6:
        return [0.0, 0.1]
    lb = value * (1.0 - perturbation)
    ub = value * (1.0 + perturbation)
    return [max(0.0, lb), ub]


def define_sensitivity_problem_rand(num_psites, values):
    """
    Define the Morris sensitivity problem for the random (randmod) model.

    Parameters are taken from get_num_params/get_param_names for 'randmod'.
    """
    num_vars = get_num_params("randmod", num_psites)
    param_names = get_param_names(num_psites, "randmod")

    if len(values) != num_vars:
        raise ValueError(f"Length mismatch: got {len(values)} values, expected {num_vars}")

    bounds = [compute_bound(v) for v in values]

    return {
        "num_vars": num_vars,
        "names": param_names,
        "bounds": bounds,
    }


def define_sensitivity_problem_ds(num_psites, values):
    """
    Define the Morris sensitivity problem for the current ODE_MODEL (dynamic-site, etc.).
    """
    num_vars = get_num_params(ODE_MODEL, num_psites)
    param_names = get_param_names(num_psites, ODE_MODEL)

    if len(values) != num_vars:
        raise ValueError(f"Length mismatch: got {len(values)} values, expected {num_vars}")

    bounds = [compute_bound(v) for v in values]

    return {
        "num_vars": num_vars,
        "names": param_names,
        "bounds": bounds,
    }


@njit(cache=True)
def _compute_Y(solution: np.ndarray, num_psites: int) -> float:
    """
    Compute scalar Y from an ODE solution according to global Y_METRIC.

    This is Numba-accelerated and operates on NumPy arrays only.
    """
    n_t = solution.shape[0]
    sum_mRNA = 0.0
    sum_protein = 0.0
    sum_sites = 0.0
    length = 2 * n_t + n_t * num_psites

    for t in range(n_t):
        sum_mRNA += solution[t, 0]
        sum_protein += solution[t, 1]
        for s in range(num_psites):
            sum_sites += solution[t, 2 + s]

    if Y_METRIC == "total_signal":
        return sum_mRNA + sum_protein + sum_sites

    if Y_METRIC == "mean_activity":
        return (sum_mRNA + sum_protein + sum_sites) / length

    if Y_METRIC == "variance":
        mean = (sum_mRNA + sum_protein + sum_sites) / length
        var_acc = 0.0
        for t in range(n_t):
            var_acc += (solution[t, 0] - mean) ** 2
            var_acc += (solution[t, 1] - mean) ** 2
            for s in range(num_psites):
                var_acc += (solution[t, 2 + s] - mean) ** 2
        return var_acc / length

    if Y_METRIC == "dynamics":
        dyn_acc = 0.0
        # mRNA chain
        prev = solution[0, 0]
        for t in range(1, n_t):
            cur = solution[t, 0]
            dyn_acc += (cur - prev) ** 2
            prev = cur
        # protein chain
        prev = solution[0, 1]
        for t in range(1, n_t):
            cur = solution[t, 1]
            dyn_acc += (cur - prev) ** 2
            prev = cur
        # sites
        for s in range(num_psites):
            prev = solution[0, 2 + s]
            for t in range(1, n_t):
                cur = solution[t, 2 + s]
                dyn_acc += (cur - prev) ** 2
                prev = cur
        return dyn_acc

    if Y_METRIC == "l2_norm":
        norm_acc = 0.0
        for t in range(n_t):
            norm_acc += solution[t, 0] ** 2
        for t in range(n_t):
            norm_acc += solution[t, 1] ** 2
        for t in range(n_t):
            for s in range(num_psites):
                norm_acc += solution[t, 2 + s] ** 2
        return math.sqrt(norm_acc)

    raise ValueError("Unknown Y_METRIC")


def _perturb_solve(i_X_tuple):
    """
    Worker: solve ODE for one perturbed parameter set and compute Y.
    This is called in parallel threads; solve_ode may internally use JAX+Diffrax.
    """
    i, X, init_cond, num_psites, time_points = i_X_tuple
    # unpack parameters: first 4 core then S-sites then D-sites
    A, B, C, D, *rest = X
    S_list = rest[:num_psites]
    D_list = rest[num_psites:]
    params = (A, B, C, D, *S_list, *D_list)
    solution, flat_psite_mRNA = solve_ode(params, init_cond, num_psites, time_points)
    # convert to NumPy for Numba function, if solve_ode returns JAX arrays
    solution_np = np.asarray(solution, dtype=np.float64)
    Y_val = _compute_Y(solution_np, num_psites)
    return i, solution_np, flat_psite_mRNA, Y_val


def _sensitivity_analysis(pr_data, p_data, rna_data, popt, time_points, num_psites,
                          psite_labels, state_labels, init_cond, gene):
    """
    Perform Morris sensitivity analysis using the current ODE model.

    Uses thread-based parallelism so it is compatible with JAX/Diffrax in solve_ode.
    """

    # Build SALib problem definition
    if ODE_MODEL == "randmod":
        problem = define_sensitivity_problem_rand(num_psites=num_psites, values=popt)
    else:
        problem = define_sensitivity_problem_ds(num_psites=num_psites, values=popt)

    N = NUM_TRAJECTORIES
    num_levels = PARAMETER_SPACE
    param_values = morris.sample(problem, N=N, num_levels=num_levels, local_optimization=True)
    n_samples = len(param_values)

    Y = np.zeros(n_samples, dtype=np.float64)

    all_model_psite_solutions = np.zeros((n_samples, len(time_points), num_psites))
    all_protein_solutions = np.zeros((n_samples, len(time_points)))
    all_mrna_solutions = np.zeros((n_samples, len(time_points)))
    all_flat_mRNA = np.zeros((n_samples, len(TIME_POINTS_RNA)))
    trajectories_with_params = []

    tasks = [
        (i, X, init_cond, num_psites, time_points)
        for i, X in enumerate(param_values)
    ]

    logger.info(f"[{gene}]      Sensitivity Analysis started...")

    # Use threads instead of processes to avoid JAX/XLA issues
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_perturb_solve, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            i, solution, flat_psite_mRNA, Y_val = fut.result()
            Y[i] = Y_val
            all_mrna_solutions[i] = solution[:, 0]
            all_protein_solutions[i] = solution[:, 1]
            all_model_psite_solutions[i] = solution[:, 2:2 + num_psites]
            all_flat_mRNA[i] = flat_psite_mRNA[:len(TIME_POINTS_RNA)]
            trajectories_with_params.append({
                "params": param_values[i],
                "solution": solution,
                "rmse": None,
            })

    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"[{gene}]      Sensitivity Analysis completed")
    logger.info("           --------------------------------")

    Si = analyze(
        problem,
        param_values,
        Y,
        num_levels=num_levels,
        conf_level=0.99,
        scaled=True,
        print_to_console=False,
    )

    # Compare to reference data
    psite_data_ref = p_data
    protein_data_ref = pr_data.reshape(-1)
    rna_ref = rna_data.reshape(-1)

    protein_preds = all_protein_solutions[:, :]
    psite_preds = all_model_psite_solutions[:, :, :]
    rna_preds = all_mrna_solutions[:, -len(TIME_POINTS_RNA):]

    rna_diff = np.abs(rna_preds - rna_ref[np.newaxis, :]) / rna_ref.size
    psite_diff = np.abs(psite_preds - psite_data_ref.T[np.newaxis, :, :]) / psite_data_ref.size
    protein_diff = np.abs(protein_preds - protein_data_ref[np.newaxis, :]) / protein_data_ref.size

    rna_mse = np.mean(rna_diff ** 2, axis=1)
    psite_mse = np.mean(psite_diff ** 2, axis=(1, 2))
    protein_mse = np.mean(protein_diff ** 2, axis=1)
    rmse = np.sqrt((rna_mse + psite_mse + protein_mse) / 2.0)

    for i in range(n_samples):
        trajectories_with_params[i]["rmse"] = rmse[i]

    # Select top K trajectories
    K = int(np.ceil(NUM_TRAJECTORIES * 10 / PARAMETER_SPACE))
    best_idxs = np.argsort(rmse)[:K]
    best_trajectories = [trajectories_with_params[i] for i in best_idxs]

    best_model_psite_solutions = all_model_psite_solutions[best_idxs]
    best_mrna_solutions = all_mrna_solutions[best_idxs]
    best_protein_solutions = all_protein_solutions[best_idxs]

    n_sites = best_model_psite_solutions.shape[2]

    all_states = np.stack(
        [best_mrna_solutions, best_protein_solutions]
        + [best_model_psite_solutions[:, :, i] for i in range(n_sites)],
        axis=-1,
    )

    cutoff_idx = 8

    # True model fit with estimated parameters
    model_fit, _ = solve_ode(popt, init_cond, num_psites, time_points)

    plotter = Plotter(gene, OUT_DIR)
    plotter.plot_time_state_grid(all_states, time_points, state_labels)
    plotter.plot_phase_space(all_states, state_labels)
    plotter.plot_model_perturbations(
        problem,
        Si,
        cutoff_idx,
        time_points,
        n_sites,
        best_model_psite_solutions,
        best_mrna_solutions,
        best_protein_solutions,
        psite_labels,
        protein_data_ref,
        psite_data_ref,
        rna_ref,
        model_fit,
    )

    return Si, best_trajectories