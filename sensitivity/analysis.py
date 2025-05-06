import math
import os
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from SALib.sample import morris
from SALib.analyze.morris import analyze
from numba import njit

from config.constants import ODE_MODEL, NUM_TRAJECTORIES, PARAMETER_SPACE, TIME_POINTS_RNA, PERTURBATIONS_VALUE, \
    OUT_DIR, Y_METRIC
from config.helpers import get_number_of_params_rand, get_param_names_rand
from models import solve_ode
from plotting.plotting import Plotter
from config.logconf import setup_logger

logger = setup_logger()


def compute_bound(value, perturbation=PERTURBATIONS_VALUE):
    """
    Computes the lower and upper bounds for a given parameter value for sensitivity analysis and perturbations.

    Args:
        value (float): The parameter value.
        perturbation (float): The perturbation factor.

    Returns:
        list: A list containing the lower and upper bounds.
    """
    if abs(value) < 1e-6:
        return [0.0, 0.1]  # fallback for near-zero
    lb = value * (1 - perturbation)
    ub = value * (1 + perturbation)
    return [max(0.0, lb), ub]


def define_sensitivity_problem_rand(num_psites, values):
    """
    Defines the Morris sensitivity analysis problem for the random model.

    Args:
        num_psites (int): Number of phosphorylation sites.
        values (list): List of parameter values.

    Returns:
        dict: A dictionary containing the number of variables, parameter names, and bounds.
    """
    num_vars = get_number_of_params_rand(num_psites)
    param_names = get_param_names_rand(num_psites)

    assert len(values) == num_vars, "Length mismatch with values"

    _bounds = [compute_bound(v) for v in values]

    return {
        'num_vars': num_vars,
        'names': param_names,
        'bounds': _bounds
    }


def define_sensitivity_problem_ds(num_psites, values):
    """
    Defines the Morris sensitivity analysis problem for the dynamic-site model.

    Args:
        num_psites (int): Number of phosphorylation sites.
        values (list): List of parameter values.

    Returns:
        dict: A dictionary containing the number of variables, parameter names, and bounds.
    """
    num_vars = 4 + 2 * num_psites
    param_names = ['A', 'B', 'C', 'D'] + \
                  [f'S{i + 1}' for i in range(num_psites)] + \
                  [f'D{i + 1}' for i in range(num_psites)]

    assert len(values) == num_vars, "Length mismatch with values"

    _bounds = [compute_bound(v) for v in values]

    return {
        'num_vars': num_vars,
        'names': param_names,
        'bounds': _bounds
    }

@njit(cache=True)
def _compute_Y(solution: np.ndarray, num_psites: int) -> float:
    """
    Compute the scalar Y based on the global Y_METRIC flag.

    Args:
        solution (np.ndarray): The solution array from the ODE solver.
        num_psites (int): Number of phosphorylation sites.

    Returns:
        float: The computed Y value based on the selected metric.
    """
    n_t = solution.shape[0]
    # compute sum of mRNA and sites, and build flattened length
    sum_mRNA = 0.0
    sum_sites = 0.0
    length = n_t + n_t * num_psites

    for t in range(n_t):
        sum_mRNA += solution[t, 0]
        for s in range(num_psites):
            sum_sites += solution[t, 2 + s]

    if Y_METRIC == 'total_signal':
        return sum_mRNA + sum_sites

    # mean
    if Y_METRIC == 'mean_activity':
        return (sum_mRNA + sum_sites) / length

    # variance
    if Y_METRIC == 'variance':
        mean = (sum_mRNA + sum_sites) / length
        var_acc = 0.0
        # variance over all entries
        for t in range(n_t):
            var_acc += (solution[t, 0] - mean) ** 2
            for s in range(num_psites):
                var_acc += (solution[t, 2 + s] - mean) ** 2
        return var_acc / length

    # dynamics (sum of squared diffs)
    if Y_METRIC == 'dynamics':
        # flatten index i = t for mRNA
        # i = n_t + t * num_psites + s for sites
        prev = solution[0, 0]
        # first the mRNA chain
        dyn_acc = 0.0
        for t in range(1, n_t):
            cur = solution[t, 0]
            dyn_acc += (cur - prev) ** 2
            prev = cur
        # then site chains in sequence
        for s in range(num_psites):
            prev = solution[0, 2 + s]
            for t in range(1, n_t):
                cur = solution[t, 2 + s]
                dyn_acc += (cur - prev) ** 2
                prev = cur
        return dyn_acc

    # L2 norm
    if Y_METRIC == 'l2_norm':
        norm_acc = 0.0
        # mRNA
        for t in range(n_t):
            norm_acc += solution[t, 0] ** 2
        # sites
        for t in range(n_t):
            for s in range(num_psites):
                norm_acc += solution[t, 2 + s] ** 2
        return math.sqrt(norm_acc)

    raise ValueError("Unknown Y_METRIC")

def _perturb_solve(i_X_tuple):
    """
    Worker: solve ODE for one parameter set.

    Args:
        i_X_tuple (tuple): A tuple containing the index and parameter values.

    Returns:
        tuple: The index, solution, flat phosphorylation site mRNA, and Y value.
    """
    i, X, init_cond, num_psites, time_points = i_X_tuple
    A, B, C, D, *rest = X
    S_list = rest[:num_psites]
    D_list = rest[num_psites:]
    params = (A, B, C, D, *S_list, *D_list)
    solution, flat_psite_mRNA = solve_ode(params, init_cond, num_psites, time_points)
    Y_val = _compute_Y(solution, num_psites)
    return i, solution, flat_psite_mRNA, Y_val

def _sensitivity_analysis(data, rna_data, popt, time_points, num_psites, psite_labels, state_labels, init_cond, gene):
    """
    Performs sensitivity analysis using the Morris method for a given ODE model.

    Args:
        time_points (list or np.ndarray): Time points for the ODE simulation.
        num_psites (int): Number of phosphorylation sites in the model.
        init_cond (list or np.ndarray): Initial conditions for the ODE model.
        gene (str): Name of the gene or protein being analyzed.

    Returns:
        - Si: Sensitivity indices computed from the Morris method.
        - trajectories_with_params: List of dictionaries containing parameter sets and their corresponding solutions.
    """

    if ODE_MODEL == 'randmod':
        problem = define_sensitivity_problem_rand(num_psites=num_psites, values=popt)
    else:
        problem = define_sensitivity_problem_ds(num_psites=num_psites, values=popt)

    N = NUM_TRAJECTORIES
    num_levels = PARAMETER_SPACE
    param_values = morris.sample(problem, N=N, num_levels=num_levels, local_optimization=True)
    Y = np.zeros(len(param_values))

    # Initialize list to collect all trajectories
    all_model_psite_solutions = np.zeros((len(param_values), len(time_points), num_psites))
    all_protein_solutions = np.zeros((len(param_values), len(time_points)))
    all_mrna_solutions = np.zeros((len(param_values), len(time_points)))
    all_flat_mRNA = np.zeros((len(param_values), len(TIME_POINTS_RNA)))
    trajectories_with_params = []

    # Build the iterable of arguments for the workers
    tasks = [
        (i, X, init_cond, num_psites, time_points)
        for i, X in enumerate(param_values)
    ]

    logger.info(f"[{gene}]      Sensitivity Analysis started...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_perturb_solve, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            i, solution, flat_psite_mRNA, Y_val = fut.result()
            # Y represents the scalar model output (observable) used
            # to compute sensitivity to parameter perturbations
            # Total phosphorylation across all sites
            Y[i] = Y_val
            # Stack all collected solutions
            # (n_samples, n_timepoints, n_sites)
            all_mrna_solutions[i] = solution[:, 0]
            all_protein_solutions[i] = solution[:, 1]
            all_model_psite_solutions[i] = np.vstack(solution[:, 2:2 + num_psites])
            all_flat_mRNA[i] = flat_psite_mRNA[:len(TIME_POINTS_RNA)]
            trajectories_with_params.append({
                "params": param_values[i],
                "solution": solution,
                "rmse": None
            })

    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"[{gene}]      Sensitivity Analysis completed")
    logger.info("           --------------------------------")
    Si = analyze(problem, param_values, Y, num_levels=num_levels, conf_level=0.99,
                 scaled=True, print_to_console=False)

    # Select the closest simulations to the data
    psite_data_ref = data
    rna_ref = rna_data.reshape(-1)

    # Compute diff from concatenated flat outputs
    psite_preds = all_model_psite_solutions[:, :, :]
    rna_preds = all_mrna_solutions[:, -len(TIME_POINTS_RNA):]

    rna_diff = np.abs(rna_preds - rna_ref[np.newaxis, :]) / rna_ref.size
    psite_diff = np.abs(psite_preds - psite_data_ref.T[np.newaxis, :, :]) / psite_data_ref.size

    rna_mse = np.mean(rna_diff ** 2, axis=1)
    psite_mse = np.mean(psite_diff ** 2, axis=(1, 2))
    rmse = np.sqrt((rna_mse + psite_mse) / 2.0)

    # Attach RMSE to each stored trajectory
    for i in range(len(param_values)):
        trajectories_with_params[i]["rmse"] = rmse[i]

    # Select the top K-closest simulations ~ 250 curves
    K = int(np.ceil(NUM_TRAJECTORIES * 10 / PARAMETER_SPACE))

    # Sort the RMSE values and get the indices of the best K
    best_idxs = np.argsort(rmse)[:K]

    # Get the best trajectories to save
    best_trajectories = [trajectories_with_params[i] for i in best_idxs]

    # Restrict the trajectories to only the closest ones 
    # Best phosphorylation site solutions
    best_model_psite_solutions = all_model_psite_solutions[best_idxs]

    # Best mRNA and protein solutions
    best_mrna_solutions = all_mrna_solutions[best_idxs]
    best_protein_solutions = all_protein_solutions[best_idxs]

    # Number of phosphorylation sites
    n_sites = best_model_psite_solutions.shape[2]

    # Best model solutions stacked
    all_states = np.stack([best_mrna_solutions, best_protein_solutions] +
                          [best_model_psite_solutions[:, :, i] for i in range(n_sites)], axis=-1)

    # cut-off time point for plotting
    cutoff_idx = 8

    # True model fit with estimated parameters
    model_fit, _ = solve_ode(popt, init_cond, num_psites, time_points)

    # Plot time wise changes for each state for parameter perturbations
    Plotter(gene, OUT_DIR).plot_time_state_grid(all_states, time_points, state_labels)

    Plotter(gene, OUT_DIR).plot_phase_space(all_states, state_labels)

    # Plot best simulations
    Plotter(gene, OUT_DIR).plot_model_perturbations(problem, Si, cutoff_idx, time_points, n_sites,
                                                    best_model_psite_solutions, best_mrna_solutions,
                                                    best_protein_solutions, psite_labels, psite_data_ref,
                                                    rna_ref, model_fit)

    return Si, best_trajectories
