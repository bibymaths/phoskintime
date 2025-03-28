from itertools import combinations

# Parameter Name Generators
def get_param_names_rand(num_psites: int) -> list:
    """
    Generate parameter names for the random model.
    Format: ['A', 'B', 'C', 'D'] +
            ['S1', 'S2', ..., 'S<num_psites>'] +
            [parameter names for all combinations of dephosphorylation sites].
    """
    param_names = ['A', 'B', 'C', 'D']
    param_names += [f'S{i}' for i in range(1, num_psites + 1)]
    for i in range(1, num_psites + 1):
        for combo in combinations(range(1, num_psites + 1), i):
            param_names.append(f"D{''.join(map(str, combo))}")
    return param_names

def get_param_names_ds(num_psites: int) -> list:
    """
    Generate parameter names for distributive or successive models.
    Format: ['A', 'B', 'C', 'D'] +
            ['S1', 'S2', ..., 'S<num_psites>'] +
            ['D1', 'D2', ..., 'D<num_psites>'].
    """
    return ['A', 'B', 'C', 'D'] + [f'S{i + 1}' for i in range(num_psites)] + [f'D{i + 1}' for i in range(num_psites)]

# Label Generators
def generate_labels_rand(num_psites: int) -> list:
    """
    Generates labels for the states based on the number of phosphorylation sites for the random model.
    Returns a list with the base labels "R" and "P", followed by labels for all combinations of phosphorylated sites.
    Example for num_psites=2: ["R", "P", "P1", "P2", "P12"]
    """
    labels = ["R", "P"]
    subsets = []
    for k in range(1, num_psites + 1):
        for comb in combinations(range(1, num_psites + 1), k):
            subsets.append("P" + "".join(map(str, comb)))
    return labels + subsets

def generate_labels_ds(num_psites: int) -> list:
    """
    Generates labels for the states based on the number of phosphorylation sites for the distributive or successive models.
    Returns a list with the base labels "R" and "P", followed by labels for each individual phosphorylated state.
    Example for num_psites=2: ["R", "P", "P1", "P2"]
    """
    return ["R", "P"] + [f"P{i}" for i in range(1, num_psites + 1)]