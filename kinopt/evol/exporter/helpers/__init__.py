from collections import defaultdict


def build_genes_data(P_initial, P_init_dense, P_estimated, residuals):
    """
    Function to build a dictionary containing data for each gene.

    Args:
        P_initial (dict): Dictionary with initial parameters.
        P_init_dense (ndarray): Dense matrix of initial parameters.
        P_estimated (ndarray): Dense matrix of estimated parameters.
        residuals (ndarray): Dense matrix of residuals.
    Returns:
        genes_data (dict): Dictionary with gene data.
    """
    genes_data = defaultdict(lambda: {"psites": [], "observed": [], "estimated": [], "residuals": []})
    keys = list(P_initial.keys())
    for i, key in enumerate(keys):
        gene, psite = key  # Split the tuple into gene and psite.
        genes_data[gene]["psites"].append(psite)
        genes_data[gene]["observed"].append(P_init_dense[i, :])
        genes_data[gene]["estimated"].append(P_estimated[i, :])
        genes_data[gene]["residuals"].append(residuals[i, :])
    return genes_data
