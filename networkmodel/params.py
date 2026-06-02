"""
Parameter Management and Transformation Module.

This module handles the interface between the biological model (which requires
positive physical parameters like rate constants) and the numerical optimizer
(which operates on a flat vector of decision variables).

**Key Concepts:**
1.  **Positivity Constraint:** Biological rates cannot be negative. To enforce this
    during optimization without using hard constraints, we use a **Softplus** transformation:
    $$ P_{physical} = ln(1 + e^{\\theta_{raw}}) $$
    This maps any real number $\\theta$ to a positive value $P$.
2.  **Vectorization:** The optimizer expects a single 1D array (`theta`). This module
    packs all distinct parameter arrays ($A_i, B_i, \dots$) into this vector and
    maintains a `slices` dictionary to unpack them back.


"""

import numpy as np

from networkmodel.config import BOUNDS_CONFIG
from networkmodel.utils import inv_softplus, softplus


def init_raw_params(defaults, custom_bounds=None):
    """
    Initializes the decision vector and bounds for the optimizer.

    This function performs three main tasks:
    1.  **Transformation:** Converts default physical values to "raw" optimization space
        using the inverse softplus function.
    2.  **Flattening:** Concatenates all parameter arrays into a single 1D vector `theta0`.
    3.  **Bounds Mapping:** Transforms physical bounds (min, max) into raw space bounds.



    Args:
        defaults (dict): Dictionary of initial physical values (e.g., {'A_i': [0.1, ...], ...}).
        custom_bounds (dict, optional): Dictionary of (min, max) tuples for specific keys.
                                        Overrides the global `BOUNDS_CONFIG`.

    Returns:
        tuple: A tuple containing:
            - theta0 (np.ndarray): The initial flattened decision vector.
            - slices (dict): Mapping of parameter names to slice objects for unpacking.
            - xl (np.ndarray): Lower bounds vector in raw space.
            - xu (np.ndarray): Upper bounds vector in raw space.
    """
    if custom_bounds is None:
        custom_bounds = {}

    vecs = []  # List to hold flattened raw parameter arrays
    slices = {}  # Dictionary to store slice indices for retrieval
    bounds = []  # List of (min, max) tuples for every element
    curr = 0  # Current index pointer in the flat vector

    # 1. Iterate over array-based parameters (Genes/Proteins)
    # These are vectors of length N (number of proteins)
    for k in ["A_i", "B_i", "C_i", "D_i", "E_i", "c_k", "tf_scale", "Dp_i"]:
        # Transform Physical -> Raw
        raw = inv_softplus(np.atleast_1d(np.asarray(defaults[k], dtype=float)))
        vecs.append(raw)

        # Record the slice for this parameter group
        length = len(raw)
        slices[k] = slice(curr, curr + length)
        curr += length

        # Determine physical bounds: Priority: Custom Bounds > Global Config
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

    # 2. Assemble final vectors
    theta0 = np.concatenate(vecs)
    xl = np.array([b[0] for b in bounds], dtype=float)
    xu = np.array([b[1] for b in bounds], dtype=float)

    if theta0.shape != xl.shape or theta0.shape != xu.shape:
        raise ValueError(f"theta0/xl/xu shape mismatch: {theta0.shape}, {xl.shape}, {xu.shape}")
    if "alpha" in slices or "beta" in slices:
        raise ValueError("alpha/beta are network construction weights and must not be optimized in theta.")

    return theta0, slices, xl, xu


def unpack_params(theta, slices):
    """
    Converts the raw decision vector back to physical parameters.

    Applies the Softplus transformation:
    $$ P = \\ln(1 + e^{\\theta}) $$
    This ensures that all output parameters are strictly positive, regardless of
    the values chosen by the optimizer.

    Args:
        theta (np.ndarray): The flat decision vector from the optimizer.
        slices (dict): The slicing dictionary created by `init_raw_params`.

    Returns:
        dict: A dictionary of physical parameters keys (e.g., 'A_i', 'c_k') mapped
              to positive numpy arrays or scalars.
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
