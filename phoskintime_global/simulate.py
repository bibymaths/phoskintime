import warnings
import numpy as np
import pandas as pd
from scipy.integrate import odeint, ODEintWarning

warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", message="Excess work done on this call")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from phoskintime_global.config import MODEL, USE_CUSTOM_SOLVER
from phoskintime_global.jacspeedup import fd_jacobian_odeint, rhs_odeint, build_S_cache_into, solve_custom

if USE_CUSTOM_SOLVER:
    print("[Solver] Using Adaptive Heun Bucketed Solver")
else:
    print("[Solver] Using Scipy ODEint")

if MODEL == 0:
    print("[Model] Using Distributive Model")
elif MODEL == 1:
    print("[Model] Using Sequential Model")
elif MODEL == 2:
    print("[Model] Using Combinatorial Model")
else:
    raise ValueError(f"Unknown MODEL value in config file: {MODEL}")

def simulate_odeint(sys, t_eval, rtol, atol, mxstep):
    """
    Returns Y with shape (T, state_dim), matching your usage (like sol.y.T).
    """
    y0 = sys.y0().astype(np.float64)
    t_eval = t_eval.astype(np.float64)

    if USE_CUSTOM_SOLVER:
        xs = solve_custom(sys, y0, t_eval, rtol=rtol, atol=atol)
        return np.ascontiguousarray(xs, dtype=np.float64)

    # odeint path
    if MODEL == 2:
        build_S_cache_into(sys.S_cache, sys.W_indptr, sys.W_indices, sys.W_data, sys.kin_Kmat, sys.c_k)
        args = sys.odeint_args(sys.S_cache)
    else:
        args = sys.odeint_args()

    xs = odeint(
        rhs_odeint,
        y0,
        t_eval,
        args=args,
        Dfun=fd_jacobian_odeint,   # keep if you want; for MODEL==2 you can set None if it causes issues
        col_deriv=False,
        rtol=rtol,
        atol=atol,
        mxstep=mxstep,
    )
    return np.ascontiguousarray(xs, dtype=np.float64)


def simulate_and_measure(sys, idx, t_points_p, t_points_r):
    times = np.unique(np.concatenate([t_points_p, t_points_r]))
    Y = simulate_odeint(sys, times, rtol=1e-5, atol=1e-7, mxstep=5000)

    # baseline indices
    prot_b = int(np.where(times == 0.0)[0][0])
    rna_b = int(np.where(times == 4.0)[0][0])  # IMPORTANT

    rows_p, rows_r = [], []
    for i, p in enumerate(idx.proteins):
        st = idx.offset_y[i]
        if MODEL == 2:
            ns = int(idx.n_states[i])
            p0 = st + 1
            tot = 0.0
            for m in range(ns):
                tot += Y[:, p0 + m]
        else:
            ns = idx.n_sites[i]
            tot = Y[:, st + 1] + (Y[:, st + 2: st + 2 + ns].sum(axis=1) if ns > 0 else 0.0)
        fc_p = np.maximum(tot, 1e-9) / np.maximum(tot[prot_b], 1e-9)
        rows_p.append(pd.DataFrame({"protein": p, "time": times, "pred_fc": fc_p}))

        # RNA baseline at t=4
        R = Y[:, st]
        fc_r = np.maximum(R, 1e-9) / np.maximum(R[rna_b], 1e-9)
        rows_r.append(pd.DataFrame({"protein": p, "time": times, "pred_fc": fc_r}))

    df_p = pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame()
    df_r = pd.concat(rows_r, ignore_index=True) if rows_r else pd.DataFrame()

    df_p = df_p[df_p["time"].isin(t_points_p)]
    df_r = df_r[df_r["time"].isin(t_points_r)]
    return df_p, df_r
