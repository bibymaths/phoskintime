from collections import defaultdict

def build_genes_data(P_initial, P_init_dense, P_estimated, residuals):
    """
    Given the dictionary P_initial (with keys (gene, psite)), and the matrices
    P_init_dense, P_estimated, and residuals (with rows in the same order as list(P_initial.keys())),
    build a dictionary keyed by gene where each value is a dict with:
      - "psites": list of psite labels
      - "observed": list of observed time-series (as arrays)
      - "estimated": list of estimated time-series (as arrays)
      - "residuals": list of residuals (as arrays)
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