"""Simulate a System with Diffrax and extract protein, RNA, and phospho measurement tables; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config, networkmodel.jax_backend."""

import warnings
import numpy as np
import pandas as pd

import jax.numpy as jnp
import diffrax

from networkmodel.pinn.objective import _rebuild_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from networkmodel.config import MODEL, ODE_ABS_TOL, ODE_REL_TOL, ODE_MAX_STEPS
from networkmodel.backend import DiffraxSolverConfig, make_networkmodel_rhs, solve_diffrax


def combinatorial_site_signals_streaming(states, n_sites):
    """Return per-site combinatorial signals without building an ns x n_sites bit matrix."""
    state_view = np.asarray(states, dtype=np.float64)
    if state_view.ndim != 2:
        raise ValueError("states must be a two-dimensional time-by-state array")
    ns = state_view.shape[1]
    n_sites = int(n_sites)
    out = np.empty((state_view.shape[0], n_sites), dtype=np.float64)
    masks = np.arange(ns, dtype=np.uint64)
    for site in range(n_sites):
        weights = ((masks >> np.uint64(site)) & np.uint64(1)).astype(np.float64, copy=False)
        out[:, site] = state_view @ weights
        del weights
    return out


def simulate_diffrax(sys, t_eval, rtol=None, atol=None, max_steps=None, solver_name="Kvaerno4"):
    """Simulate a System over requested time points with Diffrax
    
    Args:
        sys: Input value used by this routine.
        t_eval: Input value used by this routine.
        rtol: Input value used by this routine.
        atol: Input value used by this routine.
        max_steps: Input value used by this routine.
        solver_name: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    y0 = np.asarray(sys.y0(), dtype=np.float64)
    cfg = DiffraxSolverConfig(
        solver_name=solver_name,
        rtol=float(ODE_REL_TOL if rtol is None else rtol),
        atol=float(ODE_ABS_TOL if atol is None else atol),
        max_steps=int(ODE_MAX_STEPS if max_steps is None else max_steps),
        root_max_steps=20,
    )
    params = (sys.c_k, sys.A_i, sys.B_i, sys.C_i, sys.D_i, sys.Dp_i, sys.E_i,
              np.asarray([sys.tf_scale], dtype=np.float64))
    rhs = getattr(sys, "_cached_jax_rhs", None)
    if rhs is None:
        rhs = make_networkmodel_rhs(sys)
        sys._cached_jax_rhs = rhs

    return np.asarray(
        solve_diffrax(
            y0,
            np.asarray(t_eval, dtype=np.float64),
            params=params,
            rhs=rhs,
            config=cfg,
        ),
        dtype=np.float64,
    )

def measure_trajectory(Y, idx, times, t_points_p, t_points_r, t_points_pho):
    """Extract protein, RNA, and phospho measurement tables from a solved trajectory."""
    Y = np.asarray(Y, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

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

            states = Y[:, p0:p0 + ns]
            tot = states.sum(axis=1)
            fc_p = np.maximum(tot, 1e-12) / np.maximum(tot[prot_b], 1e-12)
            rows_p.append(pd.DataFrame({"protein": gene, "time": times, "pred_fc": fc_p}))

            if n_sites > 0:
                masks = np.arange(ns, dtype=np.uint64)
                for s_idx, psite in enumerate(idx.sites[i]):
                    weights = ((masks >> np.uint64(s_idx)) & np.uint64(1)).astype(np.float64, copy=False)
                    sig = states @ weights
                    fc = np.maximum(sig, 1e-12) / np.maximum(sig[pho_b], 1e-12)
                    rows_pho.append(
                        pd.DataFrame(
                            {
                                "protein": gene,
                                "psite": psite,
                                "time": times,
                                "pred_fc": fc,
                            }
                        )
                    )
                    del weights, sig
                del masks
            del states

        else:
            ns = int(idx.n_sites[i])

            P0 = Y[:, st + 1]
            if ns > 0:
                P_sites = Y[:, st + 2: st + 2 + ns]
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
                    rows_pho.append(
                        pd.DataFrame(
                            {
                                "protein": gene,
                                "psite": psite,
                                "time": times,
                                "pred_fc": fc,
                            }
                        )
                    )

    df_p = pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame(columns=["protein", "time", "pred_fc"])
    df_r = pd.concat(rows_r, ignore_index=True) if rows_r else pd.DataFrame(columns=["protein", "time", "pred_fc"])
    df_pho = pd.concat(rows_pho, ignore_index=True) if rows_pho else pd.DataFrame(
        columns=["protein", "psite", "time", "pred_fc"]
    )

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

def simulate_and_measure(sys, idx, t_points_p, t_points_r, t_points_pho):
    """Simulate mechanistic System and return measured output tables."""
    times = np.unique(np.concatenate([t_points_p, t_points_r, t_points_pho]).astype(np.float64))
    Y = simulate_diffrax(sys, times, rtol=1e-5, atol=1e-7, max_steps=5000)
    return measure_trajectory(Y, idx, times, t_points_p, t_points_r, t_points_pho)

def simulate_pinn_diffrax(
    sys,
    theta,
    slices,
    pinn_config,
    pinn_spec,
    t_eval,
    rtol=None,
    atol=None,
    max_steps=None,
    solver_name="Kvaerno4",
):
    """Simulate hybrid or pure NeuralODE PINN model with trained neural parameters."""
    if pinn_config is None or pinn_spec is None or not getattr(pinn_config, "enabled", False):
        return simulate_diffrax(
            sys,
            t_eval,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            solver_name=solver_name,
        )

    mode = str(getattr(pinn_config, "mode", "off")).lower()
    if mode == "off":
        return simulate_diffrax(
            sys,
            t_eval,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            solver_name=solver_name,
        )

    theta = jnp.asarray(theta, dtype=jnp.float64)
    t_eval = jnp.asarray(t_eval, dtype=jnp.float64)
    y0 = jnp.asarray(sys.y0(), dtype=jnp.float64)

    base_theta = theta[: pinn_spec.base_size]
    nn_flat = theta[pinn_spec.nn_slice]
    nn_model = _rebuild_model(pinn_spec, nn_flat)

    mech_rhs = None
    if mode == "hybrid":
        mech_rhs = make_networkmodel_rhs(sys, slices)

    def rhs(ti, yi, args):
        base_theta_arg, nn_model_arg = args

        yi = jnp.asarray(yi, dtype=jnp.float64)
        ti = jnp.asarray(ti, dtype=jnp.float64)

        ti_scaled = ti / float(pinn_config.t_scale)
        yi_scaled = yi / float(pinn_config.y_scale)

        neural = nn_model_arg(ti_scaled, yi_scaled)

        if mode == "hybrid":
            mechanistic = mech_rhs(ti, yi, base_theta_arg)
            return mechanistic + neural

        if mode == "neuralode":
            return neural

        raise ValueError(f"Unsupported pinn mode for simulation: {mode!r}")

    cfg = DiffraxSolverConfig(
        solver_name=solver_name,
        rtol=float(ODE_REL_TOL if rtol is None else rtol),
        atol=float(ODE_ABS_TOL if atol is None else atol),
        max_steps=int(ODE_MAX_STEPS if max_steps is None else max_steps),
        root_max_steps=20,
    )

    term = diffrax.ODETerm(rhs)

    sol = diffrax.diffeqsolve(
        term,
        cfg.solver(),
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=jnp.maximum((t_eval[-1] - t_eval[0]) / jnp.maximum(t_eval.size - 1, 1), 1e-3),
        y0=y0,
        args=(base_theta, nn_model),
        saveat=diffrax.SaveAt(ts=t_eval),
        stepsize_controller=diffrax.PIDController(
            rtol=cfg.rtol,
            atol=cfg.atol,
        ),
        max_steps=cfg.max_steps,
        throw=False,
    )

    Y = jnp.asarray(sol.ys, dtype=jnp.float64)
    Y = jnp.nan_to_num(Y, nan=1e6, posinf=1e6, neginf=-1e6)

    return np.asarray(Y, dtype=np.float64)


def simulate_pinn_and_measure(
    sys,
    idx,
    theta,
    slices,
    pinn_config,
    pinn_spec,
    t_points_p,
    t_points_r,
    t_points_pho,
):
    """Simulate fitted hybrid/neuralode PINN model and return measured output tables."""
    times = np.unique(np.concatenate([t_points_p, t_points_r, t_points_pho]).astype(np.float64))

    Y = simulate_pinn_diffrax(
        sys=sys,
        theta=theta,
        slices=slices,
        pinn_config=pinn_config,
        pinn_spec=pinn_spec,
        t_eval=times,
        rtol=1e-5,
        atol=1e-7,
        max_steps=5000,
    )

    return measure_trajectory(Y, idx, times, t_points_p, t_points_r, t_points_pho)