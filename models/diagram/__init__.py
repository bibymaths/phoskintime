from models.diagram.helpers import create_random_diagram, create_distributive_diagram, create_successive_model
from config.constants import model_type

def illustrate(gene, num_sites):
    """
    Generate a phosphorylation diagram for the given gene and number of sites,
    using the specified model type. This function calls the appropriate diagram
    creation function based on model_type.

    Parameters:
      gene       : str, gene name (used as output identifier)
      num_sites  : int, number of phosphorylation sites
    """
    output_filename_prefix = f"{gene}_phospho_diagram"

    if model_type == "Random":
        create_random_diagram(gene, num_sites, output_filename_prefix)
    elif model_type == "Distributive":
        create_distributive_diagram(gene, num_sites, output_filename_prefix)
    elif model_type == "Successive":
        create_successive_model(gene, num_sites, output_filename_prefix)
    else:
        print(f"Model type '{model_type}' not recognized. Available types are: Random, Distributive, and Successive.")