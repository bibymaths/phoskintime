from itertools import combinations
from math import comb

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

def location(path: str, label: str = None) -> str:
    """
    Returns a clickable hyperlink string for supported terminals using ANSI escape sequences.

    Args:
        path (str): The file path or URL.
        label (str, optional): The display text for the link. Defaults to the path if not provided.

    Returns:
        str: A string that, when printed, shows a clickable link in terminals that support ANSI hyperlinks.
    """
    if label is None:
        label = path
    # Ensure the path is a URL (for file paths, prepend file://)
    if not (path.startswith("http://") or path.startswith("https://") or path.startswith("file://")):
        path = f"file://{path}"
    # ANSI escape sequence format: ESC ] 8 ; ; <URL> ESC \ <label> ESC ] 8 ; ; ESC \
    return f"\033]8;;{path}\033\\{label}\033]8;;\033\\"

def get_number_of_params_rand(num_psites):
    """
    Calculate the number of parameters required for the ODE system based on the number of phosphorylation sites.

    Args:
        num_psites (int): Number of phosphorylation sites (1 to 4).

    Returns:
        int: Total number of parameters.
    """
    base_params = 4
    phosphorylation_params = num_psites
    dephosphorylation_params = sum(comb(num_psites, i) for i in range(1, num_psites + 1))
    total_params = base_params + phosphorylation_params + dephosphorylation_params
    return total_params

def get_bounds_rand(num_psites, ub= 0, lower=0):
    """
    Generate bounds for the ODE parameters based on the number of phosphorylation sites.

    Args:
        num_psites (int): Number of phosphorylation sites.
        lower (float): Lower bound for parameters.
        upper (float): Upper bound for parameters.

    Returns:
        list: List of bounds as [lower, upper] for each parameter.
    """
    bounds = [[lower, ub]] * 4
    bounds += [[lower, ub]] * num_psites
    for i in range(1, num_psites + 1):
        for _ in combinations(range(1, num_psites + 1), i):
            bounds.append([lower, ub])
    return bounds
