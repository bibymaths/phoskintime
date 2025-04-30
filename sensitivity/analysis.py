import numpy as np
from SALib.sample import morris
from SALib.analyze.morris import analyze
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from config.constants import OUT_DIR, ODE_MODEL, COLOR_PALETTE, NUM_TRAJECTORIES, PARAMETER_SPACE, TIME_POINTS_RNA
from config.helpers import get_number_of_params_rand, get_param_names_rand
from models import solve_ode
from itertools import combinations
from config.logconf import setup_logger

logger = setup_logger()


def define_sensitivity_problem_rand(num_psites, bounds):
    """
    Defines the Morris sensitivity analysis problem for the random model.
    """
    num_vars = get_number_of_params_rand(num_psites)
    param_names = get_param_names_rand(num_psites)

    _bounds = [
                  list(bounds['A']),  # A
                  list(bounds['B']),  # B
                  list(bounds['C']),  # C
                  list(bounds['D']),  # D
              ] + [list(bounds['Ssite'])] * num_psites  # Ssite bounds

    # Additional bounds for random phosphorylation site combinations
    for i in range(1, num_psites + 1):
        for _ in combinations(range(1, num_psites + 1), i):
            # Dsite bounds for each combination
            _bounds.append(list(bounds['Dsite']))

    problem = {
        'num_vars': num_vars,
        'names': param_names,
        'bounds': _bounds
    }
    return problem


def define_sensitivity_problem_ds(num_psites, bounds):
    """
    Defines the Morris sensitivity analysis problem for a dynamic number of parameters.

    Args:
        num_psites (int): Number of phosphorylation sites.
        ub (float): Upper bound for the parameters.

    Returns:
        dict: Problem definition for sensitivity analysis.
    """
    num_vars = 4 + 2 * num_psites  # A, B, C, D, and S1, S2, ..., Sn, D1, D2, ..., Dn
    param_names = ['A', 'B', 'C', 'D'] + \
                  [f'S{i + 1}' for i in range(num_psites)] + \
                  [f'D{i + 1}' for i in range(num_psites)]
    _bounds = ([
                   list(bounds['A']),  # A
                   list(bounds['B']),  # B
                   list(bounds['C']),  # C
                   list(bounds['D']),  # D
               ] +
               [list(bounds['Ssite'])] * num_psites
               +
               [list(bounds['Dsite'])] * num_psites)
    problem = {
        'num_vars': num_vars,
        'names': param_names,
        'bounds': _bounds
    }
    return problem


def _sensitivity_analysis(data, rna_data, popt, optimal_params, time_points, num_psites, psite_labels, init_cond, gene):
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
        problem = define_sensitivity_problem_rand(num_psites=num_psites, bounds=optimal_params)
    else:
        problem = define_sensitivity_problem_ds(num_psites=num_psites, bounds=optimal_params)

    N = NUM_TRAJECTORIES
    num_levels = PARAMETER_SPACE
    param_values = morris.sample(problem, N=N, num_levels=num_levels, local_optimization=True) + popt
    Y = np.zeros(len(param_values))

    # Initialize list to collect all trajectories
    all_model_psite_solutions = np.zeros((len(param_values), len(time_points), num_psites))
    all_mrna_solutions = np.zeros((len(param_values), len(time_points)))
    all_flat_mRNA= np.zeros((len(param_values), len(TIME_POINTS_RNA)))
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
        psite_data = np.vstack(solution[:, 2:2 + num_psites])
        # Flatten and combine mRNA and psite_data into a 1D array
        combined = np.hstack([mRNA, psite_data.flatten()])
        # Stack all collected solutions
        # (n_samples, n_timepoints, n_sites)
        all_mrna_solutions[i] = mRNA
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

    # --- Select the closest simulations to the data  ---
    psite_data_ref = data
    rna_ref = rna_data.reshape(-1)

    # Compute diff from concatenated flat outputs
    psite_preds = all_model_psite_solutions[:, :, :]
    rna_preds = all_mrna_solutions[:, -len(TIME_POINTS_RNA):]

    rna_diff = np.abs(rna_preds - rna_ref[np.newaxis, :])/rna_ref.size
    psite_diff = np.abs(psite_preds - psite_data_ref.T[np.newaxis, :, :])/psite_data_ref.size

    rna_mse = np.mean(rna_diff ** 2, axis=1)
    psite_mse = np.mean(psite_diff ** 2, axis=(1, 2))
    rmse = np.sqrt((rna_mse + psite_mse) / 2.0)

    # Attach RMSE to each stored trajectory
    for i in range(len(param_values)):
        trajectories_with_params[i]["rmse"] = rmse[i]

    # Select the top K-closest simulations
    # About 25% of PARAMETER_SPACE
    # K = max(5, int(PARAMETER_SPACE * 0.25))
    # Minimum of 25% of PARAMETER_SPACE and 0.5% of NUM_TRAJECTORIES, clamped between 5 and 50
    K = min(50, max(5, min(int(PARAMETER_SPACE * 0.25), int(NUM_TRAJECTORIES * 0.01))))
    best_idxs = np.argsort(rmse)[:K]

    # Restrict the trajectories to only the closest ones
    best_model_psite_solutions = all_model_psite_solutions[best_idxs]
    best_mrna_solutions = all_mrna_solutions[best_idxs]

    # --- Plot all model_psite solutions ---
    n_sites = best_model_psite_solutions.shape[2]
    # cut-off time point
    cutoff_idx = 8

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    # --- Left plot: Until 9th time point ---
    ax = axes[0]
    for site_idx in range(n_sites):
        color = COLOR_PALETTE[site_idx]
        for sim_idx in range(best_model_psite_solutions.shape[0]):
            ax.plot(
                time_points[:cutoff_idx],
                best_model_psite_solutions[sim_idx, :cutoff_idx, site_idx],
                color=color,
                alpha=0.1,
                linewidth=0.5
            )
        mean_curve = np.mean(best_model_psite_solutions[:, :cutoff_idx, site_idx], axis=0)
        ax.plot(
            time_points[:cutoff_idx],
            mean_curve,
            color=color,
            linewidth=1
        )
        ax.plot(
            time_points[:cutoff_idx],
            data[site_idx, :cutoff_idx],
            marker='s',
            linestyle='--',
            color=color,
            markersize=5,
            linewidth=0.75,
            mew=0.5, mec='black',
        )
    for sim_idx in range(best_mrna_solutions.shape[0]):
        ax.plot(
            time_points[:cutoff_idx],
            best_mrna_solutions[sim_idx, :cutoff_idx],
            color='black',
            alpha=0.1,
            linewidth=0.5
        )
    mean_curve_mrna = np.mean(best_mrna_solutions[:, :cutoff_idx], axis=0)
    ax.plot(
        time_points[:cutoff_idx],
        mean_curve_mrna,
        color='black',
        linewidth=1
    )
    ax.plot(
        TIME_POINTS_RNA[:3],
        rna_ref[:3],
        marker='s',
        linestyle='--',
        color='black',
        markersize=5,
        linewidth=0.75,
        mew=0.5, mec='black',
    )

    ax.set_xlabel('Time (min)')
    ax.set_xticks(time_points[:cutoff_idx])
    ax.set_xticklabels(
        [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[:cutoff_idx]],
        rotation=45,
        fontsize=6
    )
    ax.set_ylabel('FC')
    ax.grid(True, alpha=0.05)

    # --- Right plot: From 9th time point onwards ---
    ax = axes[1]
    for site_idx in range(n_sites):
        color = COLOR_PALETTE[site_idx]
        for sim_idx in range(best_model_psite_solutions.shape[0]):
            ax.plot(
                time_points[cutoff_idx - 1:],
                best_model_psite_solutions[sim_idx, cutoff_idx - 1:, site_idx],
                color=color,
                alpha=0.1,
                linewidth=0.5
            )
        mean_curve = np.mean(best_model_psite_solutions[:, cutoff_idx - 1:, site_idx], axis=0)
        ax.plot(
            time_points[cutoff_idx - 1:],
            mean_curve,
            color=color,
            label=f'{psite_labels[site_idx]}',
            linewidth=1
        )
        ax.plot(
            time_points[cutoff_idx - 1:],
            data[site_idx, cutoff_idx - 1:],
            marker='s',
            linestyle='--',
            color=color,
            markersize=5,
            linewidth=0.75,
            mew=0.5, mec='black'
        )
    for sim_idx in range(best_mrna_solutions.shape[0]):
        ax.plot(
            time_points[cutoff_idx - 1:],
            best_mrna_solutions[sim_idx, cutoff_idx - 1:],
            color='black',
            alpha=0.1,
            linewidth=0.5
        )
    mean_curve_mrna = np.mean(best_mrna_solutions[:, cutoff_idx - 1:], axis=0)
    ax.plot(
        time_points[cutoff_idx - 1:],
        mean_curve_mrna,
        color='black',
        label='mRNA (R)',
        linewidth=1
    )
    ax.plot(
        TIME_POINTS_RNA[4:],
        rna_ref[4:],
        marker='s',
        linestyle='--',
        color='black',
        markersize=5,
        linewidth=0.75,
        mew=0.5, mec='black',
    )

    ax.set_xlabel('Time (min)')
    ax.set_xticks(time_points[cutoff_idx:])
    ax.set_xticklabels(
        [f"{int(tp)}" if tp > 1 else f"{tp}" for tp in time_points[cutoff_idx:]],
        rotation=45,
        fontsize=6
    )
    ax.grid(True, alpha=0.05)
    ax.legend()

    plt.suptitle(f'{gene}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT_DIR}/{gene}_sensitivity_.png", format='png', dpi=300)
    plt.close()

    # Absolute Mean of Elementary Effects : represents the overall importance
    # of each parameter, reflecting its sensitivity
    ## Bar Plot of mu* ##
    # Standard Deviation of Elementary Effects: High standard deviation suggests
    # that the parameter has nonlinear effects or is involved in interactions
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.bar(problem['names'], Si['mu_star'], yerr=Si['mu_star_conf'], color='skyblue')
    ax.set_title(f'{gene}')
    ax.set_ylabel('mu* (Importance)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{gene}_sensitivity_bar_plot_mu.png", format='png', dpi=300)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.bar(problem['names'], Si['sigma'], color='orange')
    ax.set_title(f'{gene}')
    ax.set_ylabel('σ (Standard Deviation)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{gene}_sensitivity_bar_plot_sigma.png", format='png', dpi=300)
    plt.close()

    ## Bar Plot of sigma ##
    # Distinguish between parameters with purely linear effects (low sigma) and
    # those with nonlinear or interaction effects (high sigma).
    # **--- Parameters with high mu* and high sigma ---**
    #           <particularly important to watch>
    ## Scatter Plot of mu* vs sigma ##
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(Si['mu_star'], Si['sigma'], color='green', s=100)
    for i, param in enumerate(problem['names']):
        ax.text(Si['mu_star'][i], Si['sigma'][i], param, fontsize=12, ha='right', va='bottom')
    ax.set_title(f'{gene}')
    ax.set_xlabel('mu* (Mean Absolute Effect)')
    ax.set_ylabel('σ (Standard Deviation)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{gene}_sensitivity_scatter_plot_musigma.png", format='png', dpi=300)
    plt.close()

    # A radial plot (also known as a spider or radar plot) can give a visual
    # overview of multiple sensitivity metrics (e.g., mu*, sigma, etc.) for
    # each parameter in a circular format.

    # Each parameter gets a spoke, and the distance from the center represents
    # the sensitivity for a given metric.
    ## Radial Plot (Spider Plot) of Sensitivity Metrics ##
    categories = problem['names']
    N_cat = len(categories)
    mu_star = Si['mu_star']
    sigma = Si['sigma']
    angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    mu_star = np.concatenate((mu_star, [mu_star[0]]))
    sigma = np.concatenate((sigma, [sigma[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, mu_star, color='skyblue', alpha=0.4, label='Mu*')
    ax.fill(angles, sigma, color='orange', alpha=0.4, label='Sigma')
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f'{gene}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.savefig(f"{OUT_DIR}/{gene}_sensitivity_radial_plot.png", format='png', dpi=300)
    plt.close()

    # Visualize the proportion of total sensitivity contributed by each
    # parameter using a pie chart, showing the relative importance of each
    # parameter's contribution to sensitivity.
    ## Pie Chart for Sensitivity Contribution ##
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(Si['mu_star'], labels=problem['names'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors,
           textprops={'fontsize': 6})
    ax.set_title(f'{gene}')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{gene}_sensitivity_pie_chart.png", format='png', dpi=300)
    plt.close()
    return Si, trajectories_with_params