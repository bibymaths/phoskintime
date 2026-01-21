import numpy as np

from global_model.config import BOUNDS_CONFIG
from global_model.utils import inv_softplus, softplus


def init_raw_params(defaults, custom_bounds=None):
    """
    Initializes decision vector and bounds for optimization.

    Args:
        defaults (dict): Initial values for parameters (physical space).
        custom_bounds (dict, optional): Dictionary of (min, max) tuples for each parameter key.
                                        If provided, overrides global BOUNDS_CONFIG.

    Returns:
        tuple: (theta0, slices, xl, xu)
    """
    if custom_bounds is None:
        custom_bounds = {}

    vecs = []
    slices = {}
    bounds = []
    curr = 0

    # 1. Iterate over array-based parameters
    for k in ["c_k", "A_i", "B_i", "C_i", "D_i", "Dp_i", "E_i"]:
        raw = inv_softplus(defaults[k])
        vecs.append(raw)
        length = len(raw)
        slices[k] = slice(curr, curr + length)
        curr += length

        # Priority: Custom Bounds > Global Config
        if k in custom_bounds:
            phys_min, phys_max = custom_bounds[k]
        else:
            phys_min, phys_max = BOUNDS_CONFIG[k]

        # Convert physical bounds to raw (softplus-inverse) space
        # We assume bounds are uniform for all elements in the vector 'k'
        raw_min = inv_softplus(np.array([phys_min]))[0]
        raw_max = inv_softplus(np.array([phys_max]))[0]

        # Extend bounds list for every element in this parameter vector
        bounds.extend([(raw_min, raw_max)] * length)

    # 2. Handle scalar tf_scale
    raw_tf = inv_softplus(np.array([defaults["tf_scale"]]))
    vecs.append(raw_tf)
    slices["tf_scale"] = slice(curr, curr + 1)

    if "tf_scale" in custom_bounds:
        phys_min, phys_max = custom_bounds["tf_scale"]
    else:
        phys_min, phys_max = BOUNDS_CONFIG["tf_scale"]

    raw_min = inv_softplus(np.array([phys_min]))[0]
    raw_max = inv_softplus(np.array([phys_max]))[0]
    bounds.append((raw_min, raw_max))

    # 3. Assemble final vectors
    theta0 = np.concatenate(vecs)
    xl = np.array([b[0] for b in bounds], dtype=float)
    xu = np.array([b[1] for b in bounds], dtype=float)

    return theta0, slices, xl, xu


def unpack_params(theta, slices):
    """
    Converts raw decision vector (theta) back to physical parameters using softplus.
    """
    return {
        "c_k": softplus(theta[slices["c_k"]]),
        "A_i": softplus(theta[slices["A_i"]]),
        "B_i": softplus(theta[slices["B_i"]]),
        "C_i": softplus(theta[slices["C_i"]]),
        "D_i": softplus(theta[slices["D_i"]]),
        "Dp_i": softplus(theta[slices["Dp_i"]]),
        "E_i": softplus(theta[slices["E_i"]]),
        "tf_scale": softplus(theta[slices["tf_scale"]])[0]
    }