
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
mpl.use("Agg")

from kinopt.local.config.constants import OUT_DIR


def plot_fits_for_gene(gene, gene_data, real_timepoints):
    """
    Function to plot the observed and estimated phosphorylation levels for a gene.
    It generates two plots:
    1. A full timepoints plot showing all timepoints.
    2. A short timepoints plot showing only the first 7 timepoints.
    The plots are saved as PNG files in the specified output directory.

    Parameters:
    gene (str): The name of the gene.
    gene_data (dict): A dictionary containing the observed and estimated phosphorylation levels for the gene.
    real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    # Get colors from Dark2 palette
    cmap = mpl.cm.get_cmap("Dark2")
    # cmap = mpl.cm.get_cmap("Set1")
    # cmap = mpl.cm.get_cmap("Set2")

    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # Full timepoints plot
    for i, psite in enumerate(gene_data["psites"]):
        axs[0].plot(real_timepoints, gene_data["observed"][i],
                    label=f"{psite}", marker='s', linestyle='--',
                    color=colors[i], alpha=0.5, markeredgecolor='black')
        axs[0].plot(real_timepoints, gene_data["estimated"][i],
                    linestyle='-', color=colors[i])
    axs[0].set_title(f"{gene}")
    axs[0].set_xlabel("Time (minutes)")
    axs[0].set_ylabel("Phosphorylation Level (FC)")
    axs[0].grid(True, alpha=0.2)
    axs[0].set_xticks(real_timepoints[9:])

    # First 7 timepoints plot
    short_timepoints = real_timepoints[:7]
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


def plot_cumulative_residuals(gene, gene_data, real_timepoints):
    """
    Function to plot the cumulative residuals for each psite of a gene.
    It generates a plot showing the cumulative residuals over time.
    The plot is saved as a PNG file in the specified output directory.

    Parameters:
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
    It generates a plot showing the autocorrelation values over time.
    The plot is saved as a PNG file in the specified output directory.

    Parameters:
    gene (str): The name of the gene.
    gene_data (dict): A dictionary containing the residuals for each psite of the gene.
    real_timepoints (list): A list of timepoints corresponding to the observed and estimated data.
    """
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plot_acf(gene_data["residuals"][i], lags=len(real_timepoints) - 1,
                 alpha=0.03, ax=plt.gca(), label=f"{psite}",)
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
    It generates a histogram showing the distribution of residuals.
    The plot is saved as a PNG file in the specified output directory.

    Parameters:
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
    It generates a QQ plot showing the quantiles of the residuals against the quantiles of a normal distribution.
    The plot is saved as a PNG file in the specified output directory.

    Parameters:
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