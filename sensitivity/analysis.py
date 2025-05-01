import numpy as np
from SALib.sample import morris
from SALib.analyze.morris import analyze
from config.constants import ODE_MODEL, NUM_TRAJECTORIES, PARAMETER_SPACE, TIME_POINTS_RNA, PERTURBATIONS_VALUE, OUT_DIR
from config.helpers import get_number_of_params_rand, get_param_names_rand
from models import solve_ode
from plotting.plotting import Plotter
from config.logconf import setup_logger

logger = setup_logger()


def compute_bound(value, perturbation=PERTURBATIONS_VALUE):
    """
    Computes the lower and upper bounds for a given parameter value.
    The bounds are computed as a percentage of the parameter value,
    with a specified perturbation value.
    The lower bound is capped at 0.0 to avoid negative values.
    The function returns a list containing the lower and upper bounds.
    :param perturbation: Fractional perturbation around the parameter value.
    :param value: The parameter value for which to compute the bounds.
    :return: A list containing the lower and upper bounds.
    """
    if abs(value) < 1e-6:
        return [0.0, 0.1]  # fallback for near-zero
    lb = value * (1 - perturbation)
    ub = value * (1 + perturbation)
    return [max(0.0, lb), ub]


def define_sensitivity_problem_rand(num_psites, values):
    """
    Defines the Morris sensitivity analysis problem for the random model.
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


def _sensitivity_analysis(data, rna_data, popt, time_points, num_psites, psite_labels, init_cond, gene):
    """
    Performs sensitivity analysis using the Morris method for a given ODE model.

    This function defines the sensitivity problem based on the ODE model type,
    generates parameter samples, evaluates the model for each sample, and computes
    sensitivity indices. It also generates various plots to visualize the results.

    Args:
        time_points (list or np.ndarray): Time points for the ODE simulation.
        num_psites (int): Number of phosphorylation sites in the model.
        init_cond (list or np.ndarray): Initial conditions for the ODE model.
        gene (str): Name of the gene or protein being analyzed.

    Returns:
        None: The function saves sensitivity analysis results and plots to the output directory.
    """

    if ODE_MODEL == 'randmod':
        problem = define_sensitivity_problem_rand(num_psites=num_psites, values=popt)
    else:
        problem = define_sensitivity_problem_ds(num_psites=num_psites, values=popt)

    N = NUM_TRAJECTORIES
    num_levels = PARAMETER_SPACE
    param_values = morris.sample(problem, N=N, num_levels=num_levels, local_optimization=True) + popt
    Y = np.zeros(len(param_values))

    # Initialize list to collect all trajectories
    all_model_psite_solutions = np.zeros((len(param_values), len(time_points), num_psites))
    all_protein_solutions = np.zeros((len(param_values), len(time_points)))
    all_mrna_solutions = np.zeros((len(param_values), len(time_points)))
    all_flat_mRNA = np.zeros((len(param_values), len(TIME_POINTS_RNA)))
    trajectories_with_params = []

    # Loop through each parameter set and solve the ODE
    for i, X in enumerate(param_values):
        A, B, C, D, *rest = X
        S_list = rest[:num_psites]
        D_list = rest[num_psites:]
        params = (A, B, C, D, *S_list, *D_list)

        solution, flat_psite_mRNA = solve_ode(params, init_cond, num_psites, time_points)

        # Y represents the scalar model output (observable) used
        # to compute sensitivity to parameter perturbations
        # Total phosphorylation across all sites
        # Sensitivity metric (pick one):

        # Total signal (recommended default)
        Y[i] = np.sum(solution[:, 0]) + np.sum(solution[:, 2:2 + num_psites])

        # # Mean level of total activity
        # Y[i] = np.mean(np.hstack([solution[:, 0], solution[:, 2:2 + num_psites].flatten()]))

        # # Variance across time + sites
        # Y[i] = np.var(np.hstack([solution[:, 0], solution[:, 2:2 + num_psites].flatten()]))

        # # Dynamics: squared changes
        # Y[i] = np.sum(np.diff(np.hstack([solution[:, 0], solution[:, 2:2 + num_psites].flatten()])) ** 2)

        # # L2 norm (magnitude)
        # Y[i] = np.linalg.norm(np.hstack([solution[:, 0], solution[:, 2:2 + num_psites].flatten()]))

        mRNA = solution[:, 0]
        protein = solution[:, 1]
        psite_data = np.vstack(solution[:, 2:2 + num_psites])

        # Stack all collected solutions
        # (n_samples, n_timepoints, n_sites)
        all_mrna_solutions[i] = mRNA
        all_protein_solutions[i] = protein
        all_model_psite_solutions[i] = psite_data
        all_flat_mRNA[i] = flat_psite_mRNA[:len(TIME_POINTS_RNA)]
        trajectories_with_params.append({
            "params": X,
            "solution": solution,
            "rmse": None
        })

    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"[{gene}] Sensitivity Analysis completed")
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

    # Select the top K-closest simulations
    # Top percentile of the RMSE values
    K = sum(rmse <= np.percentile(rmse, 1))

    # Sort the RMSE values and get the indices of the best K
    best_idxs = np.argsort(rmse)[:K]

    # Restrict the trajectories to only the closest ones
    best_model_psite_solutions = all_model_psite_solutions[best_idxs]
    best_mrna_solutions = all_mrna_solutions[best_idxs]
    best_protein_solutions = all_protein_solutions[best_idxs]
    n_sites = best_model_psite_solutions.shape[2]
    # cut-off time point
    cutoff_idx = 8

    # Plot all simulations
    Plotter(gene, OUT_DIR).plot_model_pertrubations(problem, Si, cutoff_idx, time_points, n_sites, best_model_psite_solutions,
                                     best_mrna_solutions, best_mrna_solutions, best_protein_solutions, psite_labels,
                                     psite_data_ref, rna_ref)

    return Si, trajectories_with_params
