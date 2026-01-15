import csv

import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

mpl.use("Agg")

from kinopt.local.config.constants import OUT_DIR

def format_timepoints(tp, tol=1e-9):
    """
    Format timepoints with minimal decimals:
    - integers -> no decimal
    - non-integers -> one decimal

    Args:
        tp (array-like): Timepoints (list or np.ndarray)
        tol (float): Tolerance for floating-point integer check

    Returns:
        list[str]: Formatted labels
    """
    tp = np.asarray(tp)

    labels = []
    for x in tp:
        if np.isclose(x, np.round(x), atol=tol):
            labels.append(str(int(round(x))))
        else:
            labels.append(f"{x:.1f}")
    return labels

def plot_fits_for_gene(gene, gene_data, real_timepoints):
    """
    Function to plot the observed and estimated phosphorylation levels for each psite of a gene.

    Args:
        gene (str): The name of the gene.
        gene_data (dict): A dictionary containing observed and estimated data for each psite of the gene.
        real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    # Get colors from Dark2 palette
    cmap = mpl.cm.get_cmap("Dark2")
    # cmap = mpl.cm.get_cmap("Set1")
    # cmap = mpl.cm.get_cmap("Set2")

    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # First 7 timepoints plot
    short_timepoints = real_timepoints[:7]
    for i, psite in enumerate(gene_data["psites"]):
        axs[0].plot(short_timepoints, gene_data["observed"][i][:7],
                    label=f"{psite}", marker='s', linestyle='--',
                    color=colors[i], alpha=0.5, markeredgecolor='black')
        axs[0].plot(short_timepoints, gene_data["estimated"][i][:7],
                    linestyle='-', linewidth = 2, color=colors[i])
    axs[0].set_title(f"{gene}")
    axs[0].set_xlabel("Time (minutes)")
    axs[0].grid(True, alpha=0.2)
    axs[0].set_xticks(short_timepoints)
    axs[0].set_xticklabels(format_timepoints(short_timepoints))
    axs[0].legend(title="Residue_Position", bbox_to_anchor=(1.05, 1), loc='upper left')


    # Full timepoints plot
    xt = real_timepoints[9:]
    for i, psite in enumerate(gene_data["psites"]):
        axs[1].plot(real_timepoints, gene_data["observed"][i],
                    label=f"{psite}", marker='s', linestyle='--',
                    color=colors[i], alpha=0.5, markeredgecolor='black')
        axs[1].plot(real_timepoints, gene_data["estimated"][i],
                    linestyle='-', linewidth = 2, color=colors[i])
    axs[1].set_title(f"{gene}")
    axs[1].set_xlabel("Time (minutes)")
    axs[1].set_ylabel("Phosphorylation Level (FC)")
    axs[1].grid(True, alpha=0.2)
    axs[1].set_xticks(real_timepoints[9:])
    axs[1].set_xticklabels(format_timepoints(xt))

    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_fit_.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def export_outcomes_to_csv(outcomes, csv_path):
    """
    Export multistart optimization outcomes to CSV.

    One row per start, scalar diagnostics only.
    """
    # determine best objective for deltas
    best_fun = min(o.fun for o in outcomes)

    rows = []
    for rank, o in enumerate(sorted(outcomes, key=lambda x: x.fun), start=1):
        rows.append({
            "rank": rank,
            "start_id": o.start_id,
            "seed": o.seed,
            "fun": o.fun,
            "delta_from_best": o.fun - best_fun,
            "success": int(o.success),
            "constr_violation": o.constr_violation,
            "runtime_s": o.runtime_s,
            "param_l2_norm": float(np.linalg.norm(o.optimized_params)),
        })

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def plot_cumulative_residuals(gene, gene_data, real_timepoints):
    """
    Function to plot the cumulative residuals for each psite of a gene.

    Args:
        gene (str): The name of the gene.
        gene_data (dict): A dictionary containing the residuals for each psite of the gene.
        real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plt.plot(real_timepoints, np.cumsum(gene_data["residuals"][i]),
                 label=f"{psite}", marker='o', color=colors[i],
                 alpha=0.8, markeredgecolor='black')
    plt.title(f"{gene}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Cumulative Residuals")
    plt.grid(True, alpha=0.2)
    plt.legend(title="Residue_Position")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_cumulative_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


def plot_autocorrelation_residuals(gene, gene_data, real_timepoints):
    """
    Function to plot the autocorrelation of residuals for each psite of a gene.

    Args:
        gene (str): The name of the gene.
        gene_data (dict): A dictionary containing the residuals for each psite of the gene.
        real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plot_acf(gene_data["residuals"][i], lags=len(real_timepoints) - 1,
                 alpha=0.03, ax=plt.gca(), label=f"{psite}", )
    plt.title(f"{gene}")
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_autocorrelation_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


def plot_histogram_residuals(gene, gene_data, real_timepoints):
    """
    Function to plot histograms of residuals for each psite of a gene.

    Args:
        gene (str): The name of the gene.
        gene_data (dict): A dictionary containing the residuals for each psite of the gene.
        real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        sns.histplot(gene_data["residuals"][i], bins=20, kde=True,
                     color=colors[i], label=f"{psite}", alpha=0.8)
    plt.title(f"{gene}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.2)
    plt.legend(title="Residue_Position")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_histogram_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


def plot_qqplot_residuals(gene, gene_data, real_timepoints):
    """
    Function to plot QQ plots of residuals for each psite of a gene.

    Args:
        gene (str): The name of the gene.
        gene_data (dict): A dictionary containing the residuals for each psite of the gene.
        real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        qqplot(gene_data["residuals"][i], line='s', ax=plt.gca())
    plt.title(f"{gene}")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_qqplot_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close('all')
