
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
mpl.use("Agg")

from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from pymoo.visualization.radar import Radar
from kinopt.evol.config.constants import OUT_DIR, TIME_POINTS

def plot_residuals_for_gene(gene, gene_data):
    """
    Generates and saves combined residual-related plots for one gene with all psites in the legend.

    Args:
        gene (str): Gene identifier.
        gene_data (dict): Dictionary with keys 'psites', 'observed', 'estimated', and 'residuals' containing data for all psites.
        TIME_POINTS (np.ndarray or list): Time points corresponding to the series.
    """
    # Get colors from Dark2 palette
    cmap = mpl.cm.get_cmap("Dark2")
    # cmap = mpl.cm.get_cmap("Set1")
    # cmap = mpl.cm.get_cmap("Set2")

    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # Full timepoints plot
    for i, psite in enumerate(gene_data["psites"]):
        axs[0].plot(TIME_POINTS, gene_data["observed"][i],
                    label=f"{psite}", marker='s', linestyle='--',
                    color=colors[i], alpha=0.5, markeredgecolor='black')
        axs[0].plot(TIME_POINTS, gene_data["estimated"][i],
                    linestyle='-', color=colors[i])
    axs[0].set_title(f"{gene}")
    axs[0].set_xlabel("Time (minutes)")
    axs[0].set_ylabel("Phosphorylation Level (FC)")
    axs[0].grid(True, alpha=0.2)
    axs[0].set_xticks(TIME_POINTS[9:])

    # First 7 timepoints plot
    short_timepoints = TIME_POINTS[:7]
    for i, psite in enumerate(gene_data["psites"]):
        axs[1].plot(short_timepoints, gene_data["observed"][i][:7],
                    label=f"{psite}", marker='s', linestyle='--',
                    color=colors[i], alpha=0.5, markeredgecolor='black')
        axs[1].plot(short_timepoints, gene_data["estimated"][i][:7],
                    linestyle='-', color=colors[i])
    # axs[1].set_title(f"{gene}")
    axs[1].set_xlabel("Time (minutes)")
    axs[1].grid(True, alpha=0.2)
    axs[1].set_xticks(short_timepoints)
    axs[1].legend(title="Residue_Position", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_fit_.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Cumulative Sum of Residuals
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plt.plot(
            TIME_POINTS, np.cumsum(gene_data["residuals"][i]),
            label=f"{psite}", marker='o', color=colors[i], alpha=0.8, markeredgecolor='black'
        )
    plt.title(f"{gene}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Cumulative Residuals")
    plt.grid(True)
    plt.legend(title="Residue_Position")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cumulative_residuals_{gene}.png", format='png', dpi=300)
    plt.close()

    # 3. Autocorrelation of Residuals
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plot_acf(gene_data["residuals"][i], lags=len(TIME_POINTS) - 1, alpha=0.05)
    plt.title(f"{gene}")
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/autocorrelation_residuals_{gene}.png", format='png', dpi=300)
    plt.close()

    # 4. Histogram of Residuals
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        sns.histplot(gene_data["residuals"][i], bins=20, kde=True, color=colors[i], label=f"{psite}", alpha=0.8)
    plt.title(f"{gene}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend(title="Residue_Position")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/histogram_residuals_{gene}.png", format='png', dpi=300)
    plt.close()

    # 5. QQ Plot of Residuals
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        qqplot(gene_data["residuals"][i], line='s', ax=plt.gca())
    plt.title(f"{gene}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/qqplot_residuals_{gene}.png", format='png', dpi=300)
    plt.close('all')

def opt_analyze_nsga(problem, result, F, pairs, approx_ideal,
                approx_nadir, asf_i, pseudo_i, n_evals,
                hv, hist, val, hist_cv_avg, k, igd, best_objectives,
                waterfall_df, convergence_df, alpha_values,
                beta_values):
    """
    Generates and saves various plots related to optimization results.
    This includes design space plots, objective space plots,
    convergence plots, and parameter trend plots.

    :param problem:
    :param result:
    :param F:
    :param pairs:
    :param approx_ideal:
    :param approx_nadir:
    :param asf_i:
    :param pseudo_i:
    :param n_evals:
    :param hv:
    :param hist:
    :param val:
    :param hist_cv_avg:
    :param k:
    :param igd:
    :param best_objectives:
    :param waterfall_df:
    :param convergence_df:
    :param alpha_values:
    :param beta_values:
    """
    xl, xu = problem.bounds()
    plt.figure(figsize=(8, 8))
    plt.scatter(result.X[:, 0], result.X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.savefig(f"{OUT_DIR}/design_space.png", dpi=300)
    plt.close()
    for i, (x, y) in enumerate(pairs):
        plt.figure(figsize=(8, 8))
        plt.scatter(F[:, x], F[:, y], s=30, facecolors='none', edgecolors='blue')
        plt.title(f"Objective Space (F[{x}] vs F[{y}])")
        plt.xlabel(f"F[{x}]")
        plt.ylabel(f"F[{y}]")
        plt.savefig(f"{OUT_DIR}/objective_space_{x}_{y}.png", dpi=300)
        plt.close()
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=30, c='blue', label="Solutions")
    ax.scatter(approx_ideal[0], approx_ideal[1], approx_ideal[2], c='red', s=100, marker="*",
               label="Ideal Point (Approx)")
    ax.scatter(approx_nadir[0], approx_nadir[1], approx_nadir[2], c='black', s=100, marker="p",
               label="Nadir Point (Approx)")
    ax.set_title("Objective Space (3D)")
    ax.set_xlabel("F[0]")
    ax.set_ylabel("F[1]")
    ax.set_zlabel("F[2]")
    ax.legend()
    plt.savefig(f"{OUT_DIR}/objective_space_3d.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[asf_i, 0], F[asf_i, 1], marker="x", color="red", s=200)
    plt.title("ASF (Alpha Constraints vs Error)")
    plt.savefig(f"{OUT_DIR}/asf_plot1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 1], F[:, 2], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[asf_i, 1], F[asf_i, 2], marker="x", color="red", s=200)
    plt.title("ASF (Alpha vs Beta Constraints)")
    plt.savefig(f"{OUT_DIR}/asf_plot2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 0], F[:, 2], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[asf_i, 0], F[asf_i, 2], marker="x", color="red", s=200)
    plt.title("ASF (Beta Constraints vs Error)")
    plt.savefig(f"{OUT_DIR}/asf_plot3.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[pseudo_i, 0], F[pseudo_i, 1], marker="x", color="red", s=200)
    plt.title("Pseudo Weights (Alpha Constraints vs Error)")
    plt.savefig(f"{OUT_DIR}/pseudo_weights_plot1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 1], F[:, 2], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[pseudo_i, 1], F[pseudo_i, 2], marker="x", color="red", s=200)
    plt.title("Pseudo Weights (Alpha vs Beta Constraints)")
    plt.savefig(f"{OUT_DIR}/pseudo_weights_plot2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 0], F[:, 2], s=30, facecolors='none', edgecolors='blue')
    plt.scatter(F[pseudo_i, 0], F[pseudo_i, 2], marker="x", color="red", s=200)
    plt.title("Pseudo Weights (Beta Constraints vs Error)")
    plt.savefig(f"{OUT_DIR}/pseudo_weights_plot2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(n_evals, hv, color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/hypervolume_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(n_evals, hist_cv_avg, color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hist_cv_avg, facecolor="none", edgecolor='black', marker="p")
    plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Constraint Violation")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/convergence_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(n_evals, igd, color='black', lw=0.7, label="IGD+")
    plt.scatter(n_evals, igd, facecolor="none", edgecolor='black', marker="p")
    plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="--")
    plt.title("IGD+ Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("IGD+")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/igd_convergence.png", dpi=300)
    plt.close()

    plot = Radar(bounds=[approx_ideal, approx_nadir], normalize_each_objective=True, tight_layout=True)
    plot.add(best_objectives)
    plot.show()
    plot.save(f"{OUT_DIR}/radar_plot.png", dpi=300)

    # Get min and max values for each objective across all generations
    all_f = np.vstack([algo.opt.get("F") for algo in hist])  # Combine all generations
    min_f = np.min(all_f, axis=0)
    max_f = np.max(all_f, axis=0)
    # Set up the figure and axis for 3D plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    def update_frame(frame):
        """
        Update the frame for the animation.
        Args:
            frame (int): The current frame number.
        """
        ax.clear()
        gen_data = hist[frame].opt.get("F")  # Extract objective values for the current generation
        ax.scatter(gen_data[:, 0], gen_data[:, 1], gen_data[:, 2], c='blue', alpha=0.6)
        ax.set_title(f"Generation {frame}")
        ax.set_xlabel("F[0]")
        ax.set_ylabel("F[1]")
        ax.set_zlabel("F[2]")
        ax.set_xlim([min_f[0], max_f[0]])
        ax.set_ylim([min_f[1], max_f[1]])
        ax.set_zlim([min_f[2], max_f[2]])
    # Create the animation
    anim = FuncAnimation(fig, update_frame, frames=len(hist), repeat=False)
    # Save as GIF
    anim.save(f"{OUT_DIR}/optimization_run.gif", writer=PillowWriter(fps=10), dpi=300)

    ordered_optimizer_runs = waterfall_df.sort_values(by="Objective Value (F)", ascending=True)
    # Generate a waterfall plot for the convergence data
    plt.figure(figsize=(8, 8))
    plt.scatter(
        range(len(ordered_optimizer_runs["Objective Value (F)"])),
        ordered_optimizer_runs["Objective Value (F)"],
        color="black",
        marker="s",
        label="Objective Value"
    )

    # Customize the plot
    plt.title("")
    plt.xlabel("Optimizer Runs", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/waterfall.png', dpi=300)
    plt.close()

    # Generate a waterfall plot for the convergence data
    plt.figure(figsize=(8, 8))

    objective_values = ordered_optimizer_runs["Objective Value (F)"].values

    # Dynamically determine the threshold based on the range of objective values
    threshold = 0.05 * (objective_values.max() - objective_values.min())

    # Determine indices to plot
    indices_to_plot = [0]  # Always plot the first point
    for i in range(1, len(objective_values)):
        if abs(objective_values[i] - objective_values[i - 1]) > threshold:  # Significant change
            indices_to_plot.append(i)
        elif i % 10 == 0:  # Plot sparsely for small changes
            indices_to_plot.append(i)

            # Plot only the selected indices
    x_values = indices_to_plot
    y_values = [objective_values[i] for i in indices_to_plot]

    # Plot the line connecting points
    plt.plot(x_values, y_values, color="gray", linestyle="-", alpha=0.7)

    # Plot the points
    plt.scatter(
        x_values,
        y_values,
        color="black",
        marker="s",
        label="Objective Value"
    )

    # Customize the plot
    plt.title("")
    plt.xlabel("Optimizer Runs", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/waterfall_2.png', dpi=300)
    plt.close()

    # Covergence plot
    plt.figure(figsize=(8, 8))
    plt.bar(
        convergence_df["Iteration"],
        convergence_df["Value"].diff().fillna(convergence_df["Value"]),  # Changes in Value
        color="coral",
        alpha=0.6,
        label="∆Error"
    )
    plt.plot(convergence_df["Iteration"], convergence_df["Value"], marker="o", color="red", label="Error")
    plt.title("")
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{OUT_DIR}/convergence_2.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(val)), val, marker='o', linestyle='-', color='red')
    plt.title("")
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/convergence.png", dpi=300)
    plt.close()

    # Combine alpha and beta labels with Greek symbols
    param_labels = []

    # Add alpha labels
    for (gene, psite), kinases in alpha_values.items():
        for kinase in kinases.keys():
            param_labels.append(f"α_{gene}_{psite}_{kinase}")

    # Add beta labels
    for (kinase, psite), _ in beta_values.items():
        param_labels.append(f"β_{kinase}_{psite}")

    # Melt the DataFrame to make it long-form for easy plotting
    long_df = waterfall_df.melt(id_vars=["Individual", "Objective Value (F)"],
                                value_vars=param_labels,
                                var_name="Parameter",
                                value_name="Parameter Value")

    # Add a column to classify parameters as 'α' or 'β'
    long_df["Type"] = long_df["Parameter"].apply(
        lambda x: "α" if x.startswith("α") else ("β" if x.startswith("β") else "Other"))

    # Sort the DataFrame by "Parameter Value"
    long_df = long_df.sort_values(by="Objective Value (F)")

    plt.figure(figsize=(8, 8))
    style = {"α": {"color": 'teal', "marker": "o"},
             "β": {"color": 'indigo', "marker": "o"}}
    for param_type, props in style.items():
        subset = long_df[long_df["Type"] == param_type]
        plt.scatter(
            subset["Parameter Value"],
            subset["Objective Value (F)"],
            label=param_type,
            alpha=0.4,
            color=props["color"],
            marker=props["marker"]
        )
    plt.title("")
    plt.xlabel("Optimized Values", fontsize=10)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.legend(title="Parameter")
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/parameter_trend.png', dpi=300)
    plt.close()

    # Use a hexbin plot to visualize distributions of parameter values across the objective function
    plt.figure(figsize=(8, 8))

    hb = plt.hexbin(
        long_df["Parameter Value"],
        long_df["Objective Value (F)"],
        gridsize=50,
        cmap="viridis",
        mincnt=1
    )
    plt.colorbar(hb, label="Frequency")
    plt.title("")
    plt.xlabel("Optimized Values", fontsize=8)
    plt.ylabel("f", fontsize=10, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/parameter_scan.png", dpi=300)
    plt.close()

    # Plot the distributional plot
    plt.figure(figsize=(8, 8))  # Adjust width to accommodate many parameters
    sns.violinplot(
        x="Parameter",
        y="Parameter Value",
        hue="Objective Value (F)",  # This shows the distribution with respect to objective values
        data=long_df,
        palette="viridis",
        density_norm='width',
        cut=0,
        legend=False,
    )
    plt.xticks([])  # Remove x-axis ticks
    plt.title("")
    plt.xlabel("")  # Remove the x-axis label
    plt.ylabel("Optimized Values", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/parameter_scatter.png", format="png", dpi=300)
    plt.close()

def opt_analyze_de(long_df, convergence_df, ordered_optimizer_runs,
                   x_values, y_values, val):
    """
    Generates and saves various plots related to optimization results.
    This includes waterfall plots, convergence plots, parameter trend plots,
    and parameter scan plots.

    :param long_df:
    :param convergence_df:
    :param ordered_optimizer_runs:
    :param x_values:
    :param y_values:
    :param val:
    """
    # Waterfall plot
    plt.figure(figsize=(8, 8))
    plt.scatter(
        range(len(ordered_optimizer_runs["Objective Value (F)"])),
        ordered_optimizer_runs["Objective Value (F)"],
        color="black",
        marker="s",
        label="Objective Value"
    )
    # Customize the plot
    plt.title("")
    plt.xlabel("Optimizer Runs", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/waterfall.png', dpi=300)
    plt.close()
    # Waterfall plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_values, y_values, color="gray", linestyle="-", alpha=0.7)
    # Plot the points
    plt.scatter(
        x_values,
        y_values,
        color="black",
        marker="s",
        label="Objective Value"
    )
    plt.title("")
    plt.xlabel("Optimizer Runs", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/waterfall_2.png', dpi=300)
    plt.close()

    # Covergence plot
    plt.figure(figsize=(8, 8))
    plt.bar(
        convergence_df["Iteration"],
        convergence_df["Value"].diff().fillna(convergence_df["Value"]),  # Changes in Value
        color="coral",
        alpha=0.6,
        label="∆Error"
    )
    plt.plot(convergence_df["Iteration"], convergence_df["Value"], marker="o", color="red", label="Error")
    plt.title("")
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/convergence_2.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(val)), val, marker='o', linestyle='-', color='red')
    plt.title("")
    plt.xlabel("Iteration", fontsize=8)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/convergence.png", dpi=300)
    plt.close()
    plt.figure(figsize=(8, 8))
    style = {"α": {"color": 'teal', "marker": "o"},
             "β": {"color": 'indigo', "marker": "o"}}
    for param_type, props in style.items():
        subset = long_df[long_df["Type"] == param_type]
        plt.scatter(
            subset["Parameter Value"],
            subset["Objective Value (F)"],
            label=param_type,
            alpha=0.4,
            color=props["color"],
            marker=props["marker"]
        )
    plt.title("")
    plt.xlabel("Optimized Values", fontsize=10)
    plt.ylabel("f", fontsize=8, fontstyle='italic')
    plt.legend(title="Parameter")
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/parameter_trend.png', dpi=300)
    plt.close()
    # Use a hexbin plot to visualize distributions of parameter values
    # across the objective function
    plt.figure(figsize=(8, 8))
    hb = plt.hexbin(
        long_df["Parameter Value"],
        long_df["Objective Value (F)"],
        gridsize=50,
        cmap="viridis",
        mincnt=1
    )
    plt.colorbar(hb, label="Frequency")
    plt.title("")
    plt.xlabel("Optimized Values", fontsize=8)
    plt.ylabel("f", fontsize=10, fontstyle='italic')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/parameter_scan.png", dpi=300)
    plt.close()
    # Plot the distributional plot
    plt.figure(figsize=(8, 8))  # Adjust width to accommodate many parameters
    sns.violinplot(
        x="Parameter",
        y="Parameter Value",
        hue="Objective Value (F)",  # This shows the distribution with respect to objective values
        data=long_df,
        palette="viridis",
        density_norm='width',
        cut=0,
        legend=False,
    )
    plt.xticks([])  # Remove x-axis ticks
    plt.title("")
    plt.xlabel("")  # Remove the x-axis label
    plt.ylabel("Optimized Values", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/parameter_scatter.png", format="png", dpi=300)
    plt.close()