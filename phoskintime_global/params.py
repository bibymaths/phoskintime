import numpy as np

from phoskintime_global.config import BOUNDS_CONFIG
from phoskintime_global.utils import inv_softplus, softplus


def init_raw_params(defaults):
    vecs = []
    slices = {}
    bounds = []
    curr = 0

    for k in ["c_k", "A_i", "B_i", "C_i", "D_i", "Dp_i", "E_i"]:
        raw = inv_softplus(defaults[k])
        vecs.append(raw)
        length = len(raw)
        slices[k] = slice(curr, curr + length)
        curr += length

        phys_min, phys_max = BOUNDS_CONFIG[k]
        raw_min = inv_softplus(np.array([phys_min]))[0]
        raw_max = inv_softplus(np.array([phys_max]))[0]
        bounds.extend([(raw_min, raw_max)] * length)

    raw_tf = inv_softplus(np.array([defaults["tf_scale"]]))
    vecs.append(raw_tf)
    slices["tf_scale"] = slice(curr, curr + 1)

    phys_min, phys_max = BOUNDS_CONFIG["tf_scale"]
    raw_min = inv_softplus(np.array([phys_min]))[0]
    raw_max = inv_softplus(np.array([phys_max]))[0]
    bounds.append((raw_min, raw_max))

    theta0 = np.concatenate(vecs)
    xl = np.array([b[0] for b in bounds], dtype=float)
    xu = np.array([b[1] for b in bounds], dtype=float)
    return theta0, slices, xl, xu


def unpack_params(theta, slices):
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
