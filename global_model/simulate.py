import warnings
import numpy as np
import pandas as pd
from scipy.integrate import odeint, ODEintWarning

warnings.filterwarnings("ignore", category=ODEintWarning)
warnings.filterwarnings("ignore", message="Excess work done on this call")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from global_model.config import MODEL, USE_CUSTOM_SOLVER, RESULTS_DIR
from global_model.jacspeedup import fd_jacobian_odeint, rhs_odeint, build_S_cache_into, solve_custom

from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)

if USE_CUSTOM_SOLVER:
    logger.info("[Solver] Using Adaptive Heun Bucketed Solver")
else:
    logger.info("[Solver] Using Scipy ODEint")

if MODEL == 0:
    logger.info("[Model] Using Distributive Model")
elif MODEL == 1:
    logger.info("[Model] Using Sequential Model")
elif MODEL == 2:
    logger.info("[Model] Using Combinatorial Model")
else:
    raise ValueError(f"Unknown MODEL value in config file: {MODEL}")


def simulate_odeint(sys, t_eval, rtol, atol, mxstep):
    """
    Returns Y with shape (T, state_dim), matching your usage (like sol.y.T).
    """
    y0 = sys.y0().astype(np.float64, copy=False)
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
        Dfun=fd_jacobian_odeint, # For MODEL==2, set None if it causes issues
        col_deriv=False,
        rtol=rtol,
        atol=atol,
        mxstep=mxstep,
    )
    return np.ascontiguousarray(xs, dtype=np.float64)


def simulate_and_measure(sys, idx, t_points_p, t_points_r, t_points_pho):
    times = np.unique(np.concatenate([t_points_p, t_points_r, t_points_pho]).astype(np.float64))
    Y = simulate_odeint(sys, times, rtol=1e-5, atol=1e-7, mxstep=5000)

    def _bidx(t0: float) -> int:
        return int(np.argmin(np.abs(times - float(t0))))

    prot_b = _bidx(0.0)
    rna_b = _bidx(4.0)
    pho_b = _bidx(0.0)

    rows_p, rows_r, rows_pho = [], [], []

    for i, gene in enumerate(idx.proteins):
        st = int(idx.offset_y[i])

        R = Y[:, st]
        fc_r = np.maximum(R, 1e-12) / np.maximum(R[rna_b], 1e-12)
        rows_r.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_r}))

        if MODEL == 2:
            ns = int(idx.n_states[i])
            n_sites = int(idx.n_sites[i])
            p0 = st + 1

            states = Y[:, p0:p0 + ns]  # (T, ns)
            tot = states.sum(axis=1)  # (T,)
            fc_p = np.maximum(tot, 1e-12) / np.maximum(tot[prot_b], 1e-12)
            rows_p.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_p}))

            if n_sites > 0:
                m = np.arange(ns, dtype=np.uint32)[:, None]
                j = np.arange(n_sites, dtype=np.uint32)[None, :]
                bits = ((m >> j) & 1).astype(np.float64)  # (ns, n_sites)
                pho_sites = states @ bits  # (T, n_sites)

                for s_idx, psite in enumerate(idx.sites[i]):
                    sig = pho_sites[:, s_idx]
                    fc = np.maximum(sig, 1e-12) / np.maximum(sig[pho_b], 1e-12)
                    rows_pho.append(pd.DataFrame({
                        "protein": gene, "psite": psite, "time": times, "pred_fc": fc
                    }))

        else:
            ns = int(idx.n_sites[i])

            P0 = Y[:, st + 1]
            if ns > 0:
                P_sites = Y[:, st + 2: st + 2 + ns]  # (T, ns)
                pho_total = P_sites.sum(axis=1)
            else:
                P_sites = None
                pho_total = np.zeros_like(P0)

            tot = P0 + pho_total
            fc_p = np.maximum(tot, 1e-12) / np.maximum(tot[prot_b], 1e-12)
            rows_p.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_p}))

            if P_sites is not None:
                for s_idx, psite in enumerate(idx.sites[i]):
                    sig = P_sites[:, s_idx]
                    fc = np.maximum(sig, 1e-12) / np.maximum(sig[pho_b], 1e-12)
                    rows_pho.append(pd.DataFrame({
                        "protein": gene, "psite": psite, "time": times, "pred_fc": fc
                    }))

    df_p = pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame(columns=["protein", "time", "pred_fc"])
    df_r = pd.concat(rows_r, ignore_index=True) if rows_r else pd.DataFrame(columns=["protein", "time", "pred_fc"])
    df_pho = pd.concat(rows_pho, ignore_index=True) if rows_pho else pd.DataFrame(
        columns=["protein", "psite", "time", "pred_fc"])

    tp = np.asarray(t_points_p, dtype=np.float64)
    tr = np.asarray(t_points_r, dtype=np.float64)
    tph = np.asarray(t_points_pho, dtype=np.float64)

    if not df_p.empty:
        df_p = df_p[df_p["time"].isin(tp)]
    if not df_r.empty:
        df_r = df_r[df_r["time"].isin(tr)]
    if not df_pho.empty:
        df_pho = df_pho[df_pho["time"].isin(tph)]

    return df_p, df_r, df_pho
