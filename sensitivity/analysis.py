import numpy as np
from SALib.sample import morris
from SALib.analyze.morris import analyze
from matplotlib import pyplot as plt
from config.constants import OUT_DIR, ODE_MODEL
from config.helpers import (get_number_of_params_rand, get_param_names_rand,
                            get_bounds_rand)
from models import solve_ode

def define_sensitivity_problem_rand(num_psites):
    """
    Defines the Morris sensitivity analysis problem for a dynamic number of parameters.

    Args:
        num_psites (int): Number of phosphorylation sites.

    Returns:
        dict: Problem definition for sensitivity analysis.
    """
    num_vars = get_number_of_params_rand(num_psites)
    param_names = get_param_names_rand(num_psites)
    bounds = get_bounds_rand(num_psites)
    problem = {
        'num_vars': num_vars,
        'names': param_names,
        'bounds': bounds
    }
    return problem

def define_sensitivity_problem_ds(ub, num_psites):
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
    bounds = [
                 [0, ub],  # A
                 [0, ub],  # B
                 [0, ub],  # C
                 [0, ub],  # D
             ] + [[0, ub]] * num_psites + [[0, ub]] * num_psites  # S and D parameters
    problem = {
        'num_vars': num_vars,
        'names': param_names,
        'bounds': bounds
    }
    return problem

def sensitivity_analysis(time_points, num_psites, init_cond, gene):
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
        problem = define_sensitivity_problem_rand(num_psites=num_psites)
    else:
        problem = define_sensitivity_problem_ds(num_psites=num_psites)
    N = 10000
    num_levels = 400
    param_values = morris.sample(problem, N=N, num_levels=num_levels, local_optimization=True)
    Y = np.zeros(len(param_values))
    for i, X in enumerate(param_values):
        A, B, C, D, *rest = X
        S_list = rest[:num_psites]
        D_list = rest[num_psites:]
        params = (A, B, C, D, *S_list, *D_list)
        try:
            sol, _ = solve_ode(params, init_cond, num_psites, time_points)
            # Sum last time point P1, P2, ..., Pn
            Y[i] = np.sum(sol[-1, list(range(2, 2 + num_psites))])  \
                if ODE_MODEL == 'randmod'  \
                else np.sum(sol[-1, 2:2 + num_psites])
        except Exception:
            Y[i] = np.nan
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"\n\nSensitivity Analysis for Protein: {gene}\n")
    Si = analyze(problem, param_values, Y, num_levels=num_levels, conf_level=0.95, scaled=True, print_to_console=True)

    # Absolute Mean of Elementary Effects : represents the overall importance
    # of each parameter, reflecting its sensitivity
    ## Bar Plot of mu* ##
    # Standard Deviation of Elementary Effects: High standard deviation suggests
    # that the parameter has nonlinear effects or is involved in interactions
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.bar(problem['names'], Si['mu_star'], yerr=Si['mu_star_conf'], color='skyblue')
    ax.set_title(f'{gene}')
    ax.set_ylabel('mu* (Importance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/bar_plot_mu_{gene}.png", format='png', dpi=300)
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.bar(problem['names'], Si['sigma'], color='orange')
    ax.set_title(f'{gene}')
    ax.set_ylabel('σ (Standard Deviation)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/bar_plot_sigma_{gene}.png", format='png', dpi=300)
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
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/scatter_plot_musigma_{gene}.png", format='png', dpi=300)
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
    plt.savefig(f"{OUT_DIR}/radial_plot_{gene}.png", format='png', dpi=300)
    plt.close()

    # CDF can show how often the effects of certain parameters are strong or
    # weak across the model outputs.
    # Visualizing how many times a parameter has a strong effect across
    # different sample runs.
    ## Cumulative Distribution Function (CDF) of Sensitivity Indices ##
    plt.figure(figsize=(8, 8))
    for i, param in enumerate(problem['names']):
        plt.plot(np.sort(Si['mu_star']), np.linspace(0, 1, len(Si['mu_star'])), label=param)
    plt.title(f'{gene}')
    plt.xlabel('Sensitivity Index')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/cdf_plot_{gene}.png", format='png', dpi=300)
    plt.close()

    # Visualize the proportion of total sensitivity contributed by each
    # parameter using a pie chart, showing the relative importance of each
    # parameter's contribution to sensitivity.
    ## Pie Chart for Sensitivity Contribution ##
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(Si['mu_star'], labels=problem['names'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors,
           textprops={'fontsize': 8})
    ax.set_title(f'{gene}')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pie_chart_{gene}.png", format='png', dpi=300)
    plt.close()