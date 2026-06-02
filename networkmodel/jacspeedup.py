"""
Numerical Solver Wrappers and JIT Kernels.

This module acts as the high-performance computational core for the simulation.
It bridges the gap between the high-level `System` definitions and the low-level
numerical integrators (Runge-Kutta methods).

Key responsibilities:
1.  **Solver Interface**: Provides `solve_custom` to dispatch the correct integration
    routine based on the selected kinetic model (Distributive, Sequential, etc.).
2.  **JIT Compilation**: Uses Numba to compile time-critical functions (RHS evaluations,
    matrix-vector multiplications) into machine code for speed.
3.  **Sparse Matrix Operations**: Implements fast custom kernels for calculating signaling
    inputs ($S = W \cdot K$) and transcriptional regulation ($TF_{in} = A_{tf} \cdot P$).
4.  **Jacobian Approximation**: Provides finite-difference routines for estimating
    the Jacobian matrix, which is essential for stiff solvers (though mostly used here for
    diagnostics or implicit stepping if enabled).


"""

import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def build_S_cache_into(S_out, W_indptr, W_indices, W_data, kin_Kmat, c_k):
    """
    Pre-computes the signaling drive 'S' for every site at every time bucket.

    For Model 2, calculating $S = W \cdot (K(t) \cdot c_k)$ at every micro-step
    is too slow. Instead, since $K(t)$ is discretized into buckets, we can
    pre-calculate the resulting S for each bucket.

    Parallelized using `prange` for performance.

    Args:
        S_out (np.ndarray): Output cache (n_sites, n_time_buckets).
        W_*: CSR arrays for the kinase-substrate interaction matrix.
        kin_Kmat (np.ndarray): Kinase activity profiles (n_kinases, n_buckets).
        c_k (np.ndarray): Optimized kinase activity multipliers.
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
    """
    Evaluates kinase activity at time t using step interpolation (Nearest Neighbor / Bucket).

    Args:
        t (float): Current simulation time.
        grid (np.ndarray): Time grid boundaries.
        Kmat (np.ndarray): Kinase data matrix.

    Returns:
        np.ndarray: Vector of kinase activities at time t.
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