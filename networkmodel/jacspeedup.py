"""Provide small NumPy-compatible helper kernels for phosphorylation-rate cache evaluation; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on no other networkmodel modules."""

import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def build_S_cache_into(S_out, W_indptr, W_indices, W_data, kin_Kmat, c_k):
    """Fill a phosphorylation-rate cache array
    
    Args:
        S_out: Input value used by this routine.
        W_indptr: Input value used by this routine.
        W_indices: Input value used by this routine.
        W_data: Input value used by this routine.
        kin_Kmat: Input value used by this routine.
        c_k: Input value used by this routine.
    """
    n_rows = S_out.shape[0]
    n_bins = S_out.shape[1]
    for i in prange(n_rows):
        row_start = W_indptr[i]
        row_end = W_indptr[i + 1]
        # For every time bucket b...
        for b in range(n_bins):
            s = 0.0
            # ...compute the weighted sum of kinase activities acting on site i
            for p in range(row_start, row_end):
                k = W_indices[p]
                s += W_data[p] * (kin_Kmat[k, b] * c_k[k])
            S_out[i, b] = s


@njit(cache=True, fastmath=True, nogil=True)
def kin_eval_step(t, grid, Kmat):
    """Evaluate kinase inputs at a time point
    
    Args:
        t: Input value used by this routine.
        grid: Input value used by this routine.
        Kmat: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    if t <= grid[0]:
        return Kmat[:, 0].copy()
    if t >= grid[-1]:
        return Kmat[:, -1].copy()

    # Binary search to find the correct time bucket
    j = np.searchsorted(grid, t, side="right") - 1
    if j < 0:
        j = 0
    if j >= grid.size:
        j = grid.size - 1
    return Kmat[:, j].copy()
