"""Create raw parameter vectors and unpack optimized vectors into named kinetic parameter arrays; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.utils."""

import numpy as np

from networkmodel.config import BOUNDS_CONFIG
from networkmodel.utils import inv_softplus, softplus


def init_raw_params(defaults, custom_bounds=None):
    """Initialize raw optimizer parameters, slices, bounds, and defaults
    
    Args:
        defaults: Input value used by this routine.
        custom_bounds: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
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
    """Unpack a raw optimizer vector into physical parameter arrays
    
    Args:
        theta: Input value used by this routine.
        slices: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
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
