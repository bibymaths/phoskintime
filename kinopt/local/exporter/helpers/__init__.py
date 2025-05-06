from collections import defaultdict


def build_genes_data(P_initial, P_init_dense, P_estimated, residuals):
    """
    Function to build a dictionary of genes and their corresponding data.

    Args:
        P_initial (dict): Dictionary containing initial phosphorylation data.
        P_init_dense (ndarray): Dense matrix of initial phosphorylation data.
        P_estimated (ndarray): Dense matrix of estimated phosphorylation data.
        residuals (ndarray): Dense matrix of residuals.
    Returns:
        genes_data (dict): Dictionary where each key is a gene and the value is another dictionary
                           containing psites, observed, estimated, and residuals.
    """
    genes_data = defaultdict(lambda: {"psites": [], "observed": [], "estimated": [], "residuals": []})

    keys = list(P_initial.keys())

    for i, key in enumerate(keys):

        gene, psite = key
        genes_data[gene]["psites"].append(psite)
        genes_data[gene]["observed"].append(P_init_dense[i, :])
        genes_data[gene]["estimated"].append(P_estimated[i, :])
        genes_data[gene]["residuals"].append(residuals[i, :])

    return genes_data
