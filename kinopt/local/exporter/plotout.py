
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
    Plots the observed and estimated time-series for a gene (with multiple psites)
    using real timepoints for the x-axis.
    """
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plt.plot(real_timepoints, gene_data["observed"][i],
                 label=f"{psite}", marker='o', color=colors[i])
        plt.plot(real_timepoints, gene_data["estimated"][i],
                 label=f"{psite} (optimized)", marker='s', linestyle='--', color=colors[i])
    plt.title(f"Gene: {gene}")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Phosphorylation Level (FC)")
    plt.grid(True)
    plt.legend(title="Psite")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_fit_.png"
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_cumulative_residuals(gene, gene_data, real_timepoints):
    """
    Plots the cumulative sum of residuals for each psite of a gene.
    """
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(gene_data["psites"]))]
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plt.plot(real_timepoints, np.cumsum(gene_data["residuals"][i]),
                 label=f"{psite}", marker='o', color=colors[i],
                 alpha=0.8, markeredgecolor='black')
    plt.title(f"{gene}")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Cumulative Residuals")
    plt.grid(True)
    plt.legend(title="Residue_Position")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_cumulative_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def plot_autocorrelation_residuals(gene, gene_data, real_timepoints):
    """
    Plots the autocorrelation of residuals for each psite of a gene.
    """
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        plot_acf(gene_data["residuals"][i], lags=len(real_timepoints) - 1, alpha=0.05, ax=plt.gca())
    plt.title(f"{gene}")
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_autocorrelation_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def plot_histogram_residuals(gene, gene_data, real_timepoints):
    """
    Plots a histogram with KDE for the residuals of each psite of a gene.
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
    plt.grid(True)
    plt.legend(title="Residue_Position")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_histogram_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close() 
    
def plot_qqplot_residuals(gene, gene_data, real_timepoints):
    """
    Creates QQ plots of the residuals for each psite of a gene.
    """
    plt.figure(figsize=(8, 8))
    for i, psite in enumerate(gene_data["psites"]):
        qqplot(gene_data["residuals"][i], line='s', ax=plt.gca())
    plt.title(f"{gene}")
    plt.tight_layout()
    filename = f"{OUT_DIR}/{gene}_qqplot_residuals_.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close('all')