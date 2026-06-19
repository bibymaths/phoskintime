"""Provide JAX, Diffrax, and JAXopt utilities for scalar networkmodel simulation, multimodal loss evaluation, parameter projection, and ProjectedGradient optimization; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.config."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from typing import Mapping, Sequence

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
import diffrax

from config.config import setup_logger
from networkmodel.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)

LAYERS = ("mrna", "protein", "phospho")
ALIASES = {"rna": "mrna", "mrna": "mrna", "protein": "protein", "prot": "protein", "phospho": "phospho",
           "pho": "phospho"}


@dataclass(frozen=True)
class DataMode:
    """Describe which data layers contribute to the scalar loss"""
    available_layers: tuple[str, ...]
    data_mode: str
    fit_mrna: bool
    fit_protein: bool
    fit_phospho: bool

    @property
    def active_loss_terms(self) -> tuple[str, ...]:
        """Return names of active loss terms
        
        Returns:
            Computed result from this routine.
        """
        return tuple(f"{layer}_loss" for layer in self.available_layers)

    @property
    def skipped_loss_terms(self) -> tuple[str, ...]:
        """Return names of skipped loss terms
        
        Returns:
            Computed result from this routine.
        """
        return tuple(f"{layer}_loss" for layer in LAYERS if layer not in self.available_layers)


def ensure_jax_float64() -> bool:
    """Enable JAX float64 mode
    
    Returns:
        Computed result from this routine.
    """
    jax.config.update("jax_enable_x64", True)
    return bool(jax.config.jax_enable_x64)


def detect_data_mode(*, mrna=None, protein=None, phospho=None, loss_data: Mapping | None = None,
                     logger_obj=None) -> DataMode:
    """Detect which observed data layers are available
    
    Args:
        mrna: Input value used by this routine.
        protein: Input value used by this routine.
        phospho: Input value used by this routine.
        loss_data: Input value used by this routine.
        logger_obj: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
    """

    def has_frame(x) -> bool:
        if x is None:
            return False
        if hasattr(x, "empty"):
            return not bool(x.empty)
        arr = np.asarray(x)
        return arr.size > 0

    if loss_data is not None:
        mrna_on = int(loss_data.get("n_r", len(loss_data.get("obs_rna", [])))) > 0 and len(
            loss_data.get("obs_rna", [])) > 0
        protein_on = int(loss_data.get("n_p", len(loss_data.get("obs_prot", [])))) > 0 and len(
            loss_data.get("obs_prot", [])) > 0
        phospho_on = int(loss_data.get("n_ph", len(loss_data.get("obs_pho", [])))) > 0 and len(
            loss_data.get("obs_pho", [])) > 0
    else:
        mrna_on = has_frame(mrna)
        protein_on = has_frame(protein)
        phospho_on = has_frame(phospho)

    active = tuple(layer for layer, on in (("mrna", mrna_on), ("protein", protein_on), ("phospho", phospho_on)) if on)
    if not active:
        raise ValueError(
            "No PhosKinTime data layers were detected. Provide at least one of mRNA, protein, or phospho data.")
    mode = DataMode(active, "+".join(active), mrna_on, protein_on, phospho_on)
    log = logger_obj or logger
    log.info("[DataMode] Detected data mode: %s", mode.data_mode)
    log.info("[DataMode] Available layers: %s", ", ".join(mode.available_layers))
    log.info("[Objective] Active loss terms: %s", ", ".join(mode.active_loss_terms))
    log.info("[Objective] Skipped loss terms: %s", ", ".join(mode.skipped_loss_terms) or "none")
    return mode


def validate_loss_data(loss_data: Mapping, mode: DataMode) -> None:
    """Validate loss-array presence, shape, and finiteness
    
    Args:
        loss_data: Input value used by this routine.
        mode: Input value used by this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
    """
    required = {
        "protein": ("p_prot", "t_prot", "obs_prot", "w_prot"),
        "mrna": ("p_rna", "t_rna", "obs_rna", "w_rna"),
        "phospho": ("p_pho", "s_pho", "t_pho", "obs_pho", "w_pho"),
    }
    for layer in mode.available_layers:
        lengths = []
        for key in required[layer]:
            if key not in loss_data:
                raise ValueError(f"Missing required {layer} loss array '{key}'.")
            arr = np.asarray(loss_data[key])
            if arr.ndim != 1:
                raise ValueError(f"Loss array '{key}' must be one-dimensional, got shape {arr.shape}.")
            if np.any(~np.isfinite(arr.astype(float, copy=False))):
                raise ValueError(f"Loss array '{key}' contains NaN or infinite values.")
            lengths.append(arr.shape[0])
        if len(set(lengths)) != 1:
            raise ValueError(f"{layer} loss arrays have mismatched lengths: {dict(zip(required[layer], lengths))}.")


@dataclass(frozen=True)
class DiffraxSolverConfig:
    """Store Diffrax implicit-solver configuration values"""
    solver_name: str = "Kvaerno4"
    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 200000
    root_max_steps: int = 20

    def solver(self):
        """Create the configured Diffrax implicit solver
        
        Returns:
            Computed result from this routine.
        """
        name = str(self.solver_name).lower()
        if name == "kvaerno5":
            return diffrax.Kvaerno5(
                root_finder=diffrax.VeryChord(rtol=self.rtol, atol=self.atol, kappa=0.01),
                root_find_max_steps=self.root_max_steps,
            )
        return diffrax.Kvaerno4(
            root_finder=diffrax.VeryChord(rtol=self.rtol, atol=self.atol, kappa=0.01),
            root_find_max_steps=self.root_max_steps,
        )


def _default_rhs(t, y, args):
    """Handle internal default rhs"""
    rates = args
    n = y.shape[0]
    base = jnp.resize(rates, (n,))
    return base - (0.05 + jnp.abs(base)) * y


def _unpack_theta_jax(theta, slices):
    """Handle internal unpack theta jax"""
    return {
        "A_i": jax.nn.softplus(theta[slices["A_i"]]),
        "B_i": jax.nn.softplus(theta[slices["B_i"]]),
        "C_i": jax.nn.softplus(theta[slices["C_i"]]),
        "D_i": jax.nn.softplus(theta[slices["D_i"]]),
        "E_i": jax.nn.softplus(theta[slices["E_i"]]),
        "c_k": jax.nn.softplus(theta[slices["c_k"]]),
        "tf_scale": jnp.squeeze(jax.nn.softplus(theta[slices["tf_scale"]])),
        "Dp_i": jax.nn.softplus(theta[slices["Dp_i"]]),
    }


def _flatten_params_for_slices_jax(params, slices):
    """Handle internal flatten params for slices jax"""
    total = max((int(sl.stop) for sl in slices.values()), default=0)
    flat = jnp.zeros(total, dtype=jnp.float64)
    for name, sl in slices.items():
        if name not in params:
            raise ValueError(f"Missing parameter '{name}' for slice-based flattening.")
        values = jnp.ravel(jnp.asarray(params[name], dtype=jnp.float64))
        expected = int(sl.stop) - int(sl.start)
        if values.size != expected:
            raise ValueError(f"Parameter '{name}' has size {values.size}; expected {expected} from slice layout.")
        flat = flat.at[sl].set(values)
    return flat


def _defaults_vector_jax(defaults, slices):
    """Handle internal defaults vector jax"""
    if defaults is None:
        return None
    if isinstance(defaults, Mapping):
        if slices is None:
            raise ValueError("Dictionary defaults require a slice layout.")
        return _flatten_params_for_slices_jax(defaults, slices)
    return jnp.asarray(defaults, dtype=jnp.float64)


def make_networkmodel_rhs(sys, slices=None):
    """Build a JAX right-hand side for the current System topology
    
    Args:
        sys: Input value used by this routine.
        slices: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    from networkmodel.config import MODEL

    idx = sys.idx
    offsets = jnp.asarray(idx.offset_y, dtype=jnp.int32)
    site_offsets = jnp.asarray(idx.offset_s, dtype=jnp.int32)
    n_sites = jnp.asarray(idx.n_sites, dtype=jnp.int32)
    n_states = jnp.asarray(getattr(idx, "n_states", np.ones(idx.N, dtype=np.int32)), dtype=jnp.int32)
    W = jnp.asarray(sys.W_global.toarray(), dtype=jnp.float64)
    TF = jnp.asarray(sys.tf_mat.toarray(), dtype=jnp.float64)
    tf_deg = jnp.asarray(sys.tf_deg, dtype=jnp.float64)
    kin_grid = jnp.asarray(sys.kin.grid, dtype=jnp.float64)
    kin_Kmat = jnp.asarray(sys.kin.Kmat, dtype=jnp.float64)
    driver_map_np = np.full(idx.N, -1, dtype=np.int32)
    for k_name in idx.kinases:
        if k_name in idx.p2i:
            driver_map_np[idx.p2i[k_name]] = idx.k2i[k_name]
    if hasattr(idx, "proxy_map"):
        for orphan, proxy in idx.proxy_map.items():
            if orphan in idx.p2i and proxy in idx.k2i:
                driver_map_np[idx.p2i[orphan]] = idx.k2i[proxy]
    driver_map = jnp.asarray(driver_map_np, dtype=jnp.int32)
    model_id = int(MODEL)
    N = int(idx.N)
    max_sites = int(np.max(idx.n_sites)) if idx.N else 0
    max_states = int(np.max(getattr(idx, "n_states", np.ones(idx.N, dtype=np.int32)))) if idx.N else 1

    def _params(args):
        if isinstance(args, dict):
            return args
        if slices is not None:
            return _unpack_theta_jax(jnp.asarray(args, dtype=jnp.float64), slices)
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale = args
        tf_scale = jnp.ravel(tf_scale)[0]
        return {"c_k": c_k, "A_i": A_i, "B_i": B_i, "C_i": C_i, "D_i": D_i, "Dp_i": Dp_i, "E_i": E_i,
                "tf_scale": tf_scale}

    def _synth(Ai, tf_scale, u_raw):
        u = u_raw / (1.0 + jnp.abs(u_raw))
        return jnp.where(
            u >= 0.0,
            Ai * (1.0 + (tf_scale * u) / (1.0 + u + 1e-6)),
            Ai / (1.0 + tf_scale * jnp.abs(u)),
        )

    # Static metadata for compact MODEL == 2 JAX tracing.
    # Do not use global max_states/max_sites loops inside the jitted RHS.
    offsets_list = [int(x) for x in np.asarray(idx.offset_y)]
    site_offsets_list = [int(x) for x in np.asarray(idx.offset_s)]
    n_sites_list = [int(x) for x in np.asarray(idx.n_sites)]
    n_states_list = [
        int(x) for x in np.asarray(
            getattr(idx, "n_states", np.ones(idx.N, dtype=np.int32))
        )
    ]

    def rhs(t, y, args):
        par = _params(args)

        def _stepwise(row):
            idx = jnp.searchsorted(kin_grid, t, side="right") - 1
            idx = jnp.clip(idx, 0, kin_grid.shape[0] - 1)
            return row[idx]

        K_raw = jax.vmap(_stepwise)(kin_Kmat)
        Kt = K_raw * par["c_k"]
        S_all = W @ Kt

        p_vals = []
        for i in range(N):
            off = offsets[i]
            drv = driver_map[i]
            if model_id == 2:
                p0 = offsets_list[i] + 1
                nst_i = n_states_list[i]
                total_p = jnp.sum(y[p0:p0 + nst_i])
            else:
                ar = jnp.arange(max_sites, dtype=jnp.int32)
                valid = ar < n_sites[i]
                pos = jnp.minimum(off + 2 + ar, y.shape[0] - 1)
                vals = jnp.where(valid, y[pos], 0.0)
                total_p = y[off + 1] + jnp.sum(vals)
            p_vals.append(jnp.where(drv >= 0, Kt[jnp.maximum(drv, 0)], total_p))
        P_vec = jnp.stack(p_vals) if p_vals else jnp.asarray([], dtype=jnp.float64)
        TF_inputs = (TF @ P_vec) / jnp.maximum(tf_deg, 1e-12)

        dy = jnp.zeros_like(y)
        for i in range(N):
            off = offsets[i]
            s_off = site_offsets[i]
            ns = n_sites[i]
            R = y[off]
            synth = _synth(par["A_i"][i], par["tf_scale"], TF_inputs[i])
            dy = dy.at[off].set(synth - par["B_i"][i] * R)
            # if model_id == 2:
            #     # Combinatorial model: mirror models.combinatorial_rhs for the
            #     # Diffrax/JAX path. Translation feeds only mask 0, while explicit
            #     # phosphorylation transitions and implicit bit dephosphorylation
            #     # transitions move mass among all 2^n mask states.
            #     p0 = off + 1
            #     nst = n_states[i]
            #     if max_states > 0:
            #         P0 = y[p0]
            #         dy = dy.at[p0].add(par["C_i"][i] * R - par["D_i"][i] * P0)
            #
            #     # Dephosphorylation and per-site phospho-state decay for masks m > 0.
            #     for m in range(1, max_states):
            #         valid_m = m < nst
            #         pos_m = jnp.minimum(p0 + m, y.shape[0] - 1)
            #         Pm = jnp.where(valid_m, y[pos_m], 0.0)
            #         dp_rate = 0.0
            #         for j in range(max_sites):
            #             bit_set = ((m >> j) & 1) != 0
            #             valid_bit = valid_m & (j < ns) & bit_set
            #             flat_j = jnp.minimum(s_off + j, par["Dp_i"].shape[0] - 1)
            #             to = m ^ (1 << j)
            #             pos_to = jnp.minimum(p0 + to, y.shape[0] - 1)
            #             flux = jnp.where(valid_bit, par["E_i"][i] * Pm, 0.0)
            #             dy = dy.at[pos_m].add(-flux)
            #             dy = dy.at[pos_to].add(flux)
            #             dp_rate = dp_rate + jnp.where(valid_bit, par["Dp_i"][flat_j] + par["D_i"][i], 0.0)
            #         dy = dy.at[pos_m].add(-dp_rate * Pm)
            #
            #     # Explicit phosphorylation transitions, generated per protein
            #     # in the same mask/site order as the historical dense arrays.
            #     for m in range(max_states):
            #         valid_m = m < nst
            #         for j in range(max_sites):
            #             bit_unset = ((m >> j) & 1) == 0
            #             valid_tr = valid_m & (j < ns) & bit_unset
            #             to = m | (1 << j)
            #             pos_frm = jnp.minimum(p0 + m, y.shape[0] - 1)
            #             pos_to = jnp.minimum(p0 + to, y.shape[0] - 1)
            #             flat_j = jnp.minimum(s_off + j, S_all.shape[0] - 1)
            #             flux = jnp.where(valid_tr, S_all[flat_j] * y[pos_frm], 0.0)
            #             dy = dy.at[pos_frm].add(-flux)
            #             dy = dy.at[pos_to].add(flux)
            if model_id == 2:
                # Vectorized combinatorial MODEL == 2 RHS.
                # This avoids Python loops over max_states inside JAX tracing.
                p0 = offsets_list[i] + 1
                s_off_i = site_offsets_list[i]
                ns_i = n_sites_list[i]
                nst_i = n_states_list[i]

                P = y[p0:p0 + nst_i]
                states = jnp.arange(nst_i, dtype=jnp.int32)

                dy_block = jnp.zeros_like(P)

                # Translation feeds only mask 0.
                dy_block = dy_block.at[0].add(
                    par["C_i"][i] * R - par["D_i"][i] * P[0]
                )

                # Site-wise vectorized transitions.
                # Keep loop over sites only; do not loop over states in Python.
                for j in range(ns_i):
                    bit = np.int32(1 << j)
                    flat_j = s_off_i + j

                    bit_set = (states & bit) != 0
                    bit_unset = ~bit_set

                    # Dephosphorylation transition:
                    # m -> m ^ bit for states where bit is set.
                    from_set = jnp.where(bit_set, P, 0.0)
                    to_clear = states ^ bit
                    flux_back = par["E_i"][i] * from_set

                    dy_block = dy_block - flux_back
                    dy_block = dy_block.at[to_clear].add(flux_back)

                    # Per-site phospho-state decay for states where bit is set.
                    decay = (par["Dp_i"][flat_j] + par["D_i"][i]) * from_set
                    dy_block = dy_block - decay

                    # Phosphorylation transition:
                    # m -> m | bit for states where bit is unset.
                    from_unset = jnp.where(bit_unset, P, 0.0)
                    to_set = states | bit
                    flux_fwd = S_all[flat_j] * from_unset

                    dy_block = dy_block - flux_fwd
                    dy_block = dy_block.at[to_set].add(flux_fwd)

                dy = dy.at[p0:p0 + nst_i].add(dy_block)
            else:
                P = y[off + 1]
                ar = jnp.arange(max_sites, dtype=jnp.int32)
                valid = ar < ns
                site_pos = jnp.minimum(off + 2 + ar, y.shape[0] - 1)
                flat_pos = jnp.minimum(s_off + ar, S_all.shape[0] - 1)
                site_y = jnp.where(valid, y[site_pos], 0.0)
                s_rates = jnp.where(valid, S_all[flat_pos], 0.0)
                dp = jnp.where(valid, par["Dp_i"][flat_pos], 0.0)
                if model_id == 4:
                    fwd = (s_rates * P) / (1.0 + P)
                    trans = (par["C_i"][i] * R) / (1.0 + R)
                else:
                    fwd = s_rates * P
                    trans = par["C_i"][i] * R
                back = par["E_i"][i] * site_y
                site_dy = fwd - (par["E_i"][i] + dp + par["D_i"][i]) * site_y
                dy = dy.at[site_pos].add(jnp.where(valid, site_dy, 0.0))
                dy = dy.at[off + 1].set(trans - par["D_i"][i] * P - jnp.sum(fwd) + jnp.sum(back))
        return dy

    return rhs


def solve_diffrax(y0, t_eval, params=None, rhs=None, config: DiffraxSolverConfig | None = None):
    """Solve an ODE trajectory with Diffrax
    
    Args:
        y0: Input value used by this routine.
        t_eval: Input value used by this routine.
        params: Input value used by this routine.
        rhs: Input value used by this routine.
        config: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
        RuntimeError: When optimization or simulation fails.
    """
    ensure_jax_float64()
    cfg = config or DiffraxSolverConfig()

    if not isinstance(t_eval, jax.core.Tracer):
        ts_np = np.asarray(t_eval, dtype=np.float64)
        if ts_np.ndim != 1 or ts_np.size == 0:
            raise ValueError("t_eval must be a non-empty one-dimensional time grid.")
        if ts_np.size > 1 and np.any(np.diff(ts_np) <= 0.0):
            raise ValueError("t_eval must be strictly increasing for the Diffrax solver.")

    # Use jax.numpy throughout in the solve path; t_eval may be a tracer.
    ts = jnp.asarray(t_eval, dtype=jnp.float64)

    if ts.ndim != 1 or ts.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional time grid.")

    y0_j = jnp.asarray(y0, dtype=jnp.float64)
    if params is None:
        params = jnp.ones(max(1, y0_j.size), dtype=jnp.float64)
    if isinstance(params, (tuple, list)):
        args = tuple(jnp.asarray(x, dtype=jnp.float64) for x in params)
    elif isinstance(params, dict):
        args = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    else:
        args = jnp.asarray(params, dtype=jnp.float64)

    term = diffrax.ODETerm(rhs or _default_rhs)
    try:
        sol = diffrax.diffeqsolve(
            term,
            cfg.solver(),
            t0=ts[0],
            t1=ts[-1],
            dt0=jnp.maximum((ts[-1] - ts[0]) / jnp.maximum(ts.size - 1, 1), 1e-3),
            y0=y0_j,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol),
            max_steps=cfg.max_steps,
        )
    except Exception as exc:
        raise RuntimeError(f"Diffrax solver failed with {cfg.solver_name}: {exc}") from exc

    ys = jnp.asarray(sol.ys, dtype=jnp.float64)
    if ys.shape[0] != ts.shape[0]:
        raise ValueError(f"Diffrax returned invalid shape {ys.shape}; expected first dimension {ts.shape[0]}.")
    return ys


def _extract_offsets(prot_map):
    """Handle internal extract offsets"""
    pm = jnp.asarray(prot_map, dtype=jnp.int32)
    if pm.ndim == 1:
        # flat encoding: [offset0, count0, offset1, count1, ...]
        pm = pm.reshape(-1, 2)
    if pm.ndim != 2 or pm.shape[1] < 2:
        raise ValueError(
            f"prot_map must be shape (N, 2) with columns [offset, count], got shape {pm.shape}."
        )
    return pm[:, 0], pm[:, 1]


def _safe_fold_change(values, base_values):
    """Handle internal safe fold change"""
    return jnp.maximum(values, 1e-12) / jnp.maximum(base_values, 1e-12)


def _global_networkmodel_observable(Y, offsets, counts, n_sites, protein_idx, time_idx, *, layer, site_idx=None,
                                    base_idx=0, layout="standard", max_count=1):
    """Handle internal global networkmodel observable"""
    off = offsets[protein_idx]
    count = counts[protein_idx]
    t = time_idx
    b = jnp.asarray(base_idx, dtype=jnp.int32)

    if layer == "mrna":
        return _safe_fold_change(Y[t, off], Y[b, off])

    if layout == "combinatorial":
        ar = jnp.arange(max_count, dtype=jnp.int32)
        valid_state = ar < count
        state_positions = jnp.minimum(off + 1 + ar, Y.shape[1] - 1)
        states_t = jnp.where(valid_state, Y[t, state_positions], 0.0)
        states_b = jnp.where(valid_state, Y[b, state_positions], 0.0)
        if layer == "protein":
            return _safe_fold_change(jnp.sum(states_t), jnp.sum(states_b))
        site = jnp.asarray(0 if site_idx is None else site_idx, dtype=jnp.int32)
        valid_site = site < n_sites[protein_idx]
        bit_mask = jnp.where(valid_state, ((ar >> site) & 1).astype(jnp.float64), 0.0)
        pho_t = jnp.sum(states_t * bit_mask)
        pho_b = jnp.sum(states_b * bit_mask)
        return jnp.where(valid_site, _safe_fold_change(pho_t, pho_b), 0.0)

    # Standard/sequential global layout: mRNA at offset, unphosphorylated protein
    # at offset+1, and local phospho site s at offset+2+s.
    if layer == "protein":
        ar = jnp.arange(max_count + 1, dtype=jnp.int32)
        valid_state = ar < (count + 1)
        positions = jnp.minimum(off + 1 + ar, Y.shape[1] - 1)
        total_t = jnp.sum(jnp.where(valid_state, Y[t, positions], 0.0))
        total_b = jnp.sum(jnp.where(valid_state, Y[b, positions], 0.0))
        return _safe_fold_change(total_t, total_b)

    site = jnp.asarray(0 if site_idx is None else site_idx, dtype=jnp.int32)
    pred = Y[t, off + 2 + site]
    base = Y[b, off + 2 + site]
    return _safe_fold_change(pred, base)


def multimodal_loss_from_trajectory(Y, loss_data: Mapping, mode: DataMode, weights: Mapping[str, float] | None = None,
                                    *, networkmodel_layout: bool = False):
    """Compute weighted multimodal loss from a trajectory
    
    Args:
        Y: Input value used by this routine.
        loss_data: Input value used by this routine.
        mode: Input value used by this routine.
        weights: Input value used by this routine.
        networkmodel_layout: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    weights = weights or {}
    prot_map = jnp.asarray(loss_data["prot_map"], dtype=jnp.int32)
    offsets, counts = _extract_offsets(prot_map)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    breakdown = {}
    layout = str(loss_data.get("state_layout", "standard"))
    n_sites = jnp.asarray(loss_data.get("n_sites", counts), dtype=jnp.int32)
    max_count = int(np.max(np.asarray(loss_data["prot_map"], dtype=np.int32)[:, 1])) if len(
        loss_data["prot_map"]) else 1

    if mode.fit_protein:
        p = jnp.asarray(loss_data["p_prot"], dtype=jnp.int32)
        t = jnp.asarray(loss_data["t_prot"], dtype=jnp.int32)
        obs = jnp.asarray(loss_data["obs_prot"], dtype=jnp.float64)
        w = jnp.asarray(loss_data["w_prot"], dtype=jnp.float64)
        if networkmodel_layout:
            base_idx = int(loss_data.get("prot_base_idx", 0))
            pred = jax.vmap(lambda pi, ti: _global_networkmodel_observable(
                Y, offsets, counts, n_sites, pi, ti, layer="protein", base_idx=base_idx, layout=layout,
                max_count=max_count
            ))(p, t)
        else:
            pred = Y[t, offsets[p] + 1]
        loss = jnp.sum(w * (pred - obs) ** 2) / jnp.maximum(jnp.sum(w), 1.0)
        total = total + float(weights.get("protein", 1.0)) * loss
        breakdown["protein"] = loss
    if mode.fit_mrna:
        p = jnp.asarray(loss_data["p_rna"], dtype=jnp.int32)
        t = jnp.asarray(loss_data["t_rna"], dtype=jnp.int32)
        obs = jnp.asarray(loss_data["obs_rna"], dtype=jnp.float64)
        w = jnp.asarray(loss_data["w_rna"], dtype=jnp.float64)
        if networkmodel_layout:
            base_idx = int(loss_data.get("rna_base_idx", 0))
            pred = jax.vmap(lambda pi, ti: _global_networkmodel_observable(
                Y, offsets, counts, n_sites, pi, ti, layer="mrna", base_idx=base_idx, layout=layout,
                max_count=max_count
            ))(p, t)
        else:
            pred = Y[t, offsets[p]]
        loss = jnp.sum(w * (pred - obs) ** 2) / jnp.maximum(jnp.sum(w), 1.0)
        total = total + float(weights.get("mrna", weights.get("rna", 1.0))) * loss
        breakdown["mrna"] = loss
    if mode.fit_phospho:
        p = jnp.asarray(loss_data["p_pho"], dtype=jnp.int32)
        s = jnp.asarray(loss_data["s_pho"], dtype=jnp.int32)
        t = jnp.asarray(loss_data["t_pho"], dtype=jnp.int32)
        obs = jnp.asarray(loss_data["obs_pho"], dtype=jnp.float64)
        w = jnp.asarray(loss_data["w_pho"], dtype=jnp.float64)
        if networkmodel_layout:
            base_idx = int(loss_data.get("pho_base_idx", 0))
            pred = jax.vmap(lambda pi, si, ti: _global_networkmodel_observable(
                Y, offsets, counts, n_sites, pi, ti, layer="phospho", site_idx=si, base_idx=base_idx, layout=layout,
                max_count=max_count
            ))(p, s, t)
        else:
            pred = Y[t, offsets[p] + 2 + s]
        loss = jnp.sum(w * (pred - obs) ** 2) / jnp.maximum(jnp.sum(w), 1.0)
        total = total + float(weights.get("phospho", 1.0)) * loss
        breakdown["phospho"] = loss
    return total, breakdown


def project_simplex(x):
    """Project a vector onto the probability simplex
    
    Args:
        x: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    u = jnp.sort(x)[::-1]
    cssv = jnp.cumsum(u) - 1.0
    ind = jnp.arange(1, x.size + 1, dtype=jnp.float64)
    cond = u - cssv / ind > 0
    rho = jnp.sum(cond) - 1
    theta = cssv[rho] / (rho + 1.0)
    return jnp.maximum(x - theta, 0.0)


def project_alpha_blocks(alpha, block_ids):
    """Project alpha blocks onto per-block simplexes
    
    Args:
        alpha: Input value used by this routine.
        block_ids: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    a = jnp.asarray(alpha, dtype=jnp.float64)
    bids = np.asarray(block_ids)
    out = []
    for b in np.unique(bids):
        idxs = np.where(bids == b)[0]
        out.append((idxs, project_simplex(a[idxs])))
    res = a
    for idxs, vals in out:
        res = res.at[jnp.asarray(idxs)].set(vals)
    return res


def project_beta_blocks(beta, block_ids, lower=-4.0, upper=4.0):
    """Project beta blocks onto bounded per-block simplexes
    
    Args:
        beta: Input value used by this routine.
        block_ids: Input value used by this routine.
        lower: Input value used by this routine.
        upper: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    b = jnp.asarray(beta, dtype=jnp.float64)
    bids = np.asarray(block_ids)
    res = b
    for block in np.unique(bids):
        idxs_np = np.where(bids == block)[0]
        idxs = jnp.asarray(idxs_np)
        vals = res[idxs]
        vals = vals + (1.0 - jnp.sum(vals)) / vals.size
        vals = jnp.clip(vals, lower, upper)
        for _ in range(8):
            residual = 1.0 - jnp.sum(vals)
            room = jnp.where(residual >= 0, upper - vals, vals - lower)
            mask = room > 1e-12
            share = residual / jnp.maximum(jnp.sum(mask), 1)
            vals = jnp.where(mask, jnp.clip(vals + share, lower, upper), vals)
        res = res.at[idxs].set(vals)
    return res


def project_bounds(theta, lower, upper, fixed_mask=None, fixed_values=None):
    """Project parameters onto bounds and fixed values
    
    Args:
        theta: Input value used by this routine.
        lower: Input value used by this routine.
        upper: Input value used by this routine.
        fixed_mask: Input value used by this routine.
        fixed_values: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    """
    clipped = jnp.clip(jnp.asarray(theta, dtype=jnp.float64), jnp.asarray(lower, dtype=jnp.float64),
                       jnp.asarray(upper, dtype=jnp.float64))
    if fixed_mask is not None:
        clipped = jnp.where(jnp.asarray(fixed_mask, dtype=bool), jnp.asarray(fixed_values, dtype=jnp.float64), clipped)
    return clipped


@dataclass
class JaxoptResult:
    """Store scalar JAXopt optimization outputs"""
    X: np.ndarray
    F: np.ndarray
    objective_value: float
    params: np.ndarray
    state: object
    data_mode: DataMode
    loss_breakdown: dict
    optimizer: str = "jaxopt.ProjectedGradient"


def optimize_scalar_objective(objective_fun, theta0, lower, upper, *, maxiter=20000, tol=1e-6, fixed_mask=None,
                              fixed_values=None, logger_obj=None, verbose=1):
    """Optimize a scalar objective with jaxopt.ProjectedGradient
    
    Args:
        objective_fun: Input value used by this routine.
        theta0: Input value used by this routine.
        lower: Input value used by this routine.
        upper: Input value used by this routine.
        maxiter: Input value used by this routine.
        tol: Input value used by this routine.
        fixed_mask: Input value used by this routine.
        fixed_values: Input value used by this routine.
        logger_obj: Input value used by this routine.
        verbose: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
        RuntimeError: When optimization or simulation fails.
    """
    ensure_jax_float64()
    log = logger_obj or logger
    log.info("[Optimizer] Selected optimizer backend: jaxopt.ProjectedGradient")
    log.info(
        "[Constraints] Kinetic parameters use bound projection; fixed parameters are restored after each projection.")
    lower_j = jnp.asarray(lower, dtype=jnp.float64)
    upper_j = jnp.asarray(upper, dtype=jnp.float64)
    fixed_mask_j = None if fixed_mask is None else jnp.asarray(fixed_mask, dtype=bool)
    fixed_values_j = None if fixed_values is None else jnp.asarray(fixed_values, dtype=jnp.float64)

    def projection(x, hyperparams):
        lo, hi = hyperparams
        return project_bounds(x, lo, hi, fixed_mask_j, fixed_values_j)

    # Wrap objective to track best value and detect oscillation via a
    # decreasing stepsize schedule. fun_min_decrease_factor stops iteration
    # when the relative improvement over a window falls below the threshold,
    # which is exactly the oscillation / plateau signature.
    solver = jaxopt.ProjectedGradient(
        objective_fun, projection,
        value_and_grad=False,
        has_aux=False,
        stepsize=0.0,  # 0.0 = backtracking line search
        maxiter=maxiter,
        maxls=30,  # max line search steps per iteration
        tol=tol,
        verbose=1,
        implicit_diff=True,
        jit=True,
    )
    init = project_bounds(theta0, lower_j, upper_j, fixed_mask_j, fixed_values_j)
    try:
        params, state = solver.run(init, hyperparams_proj=(lower_j, upper_j))
    except Exception as exc:
        msg = f"JAXopt failed: {exc}"
        log.error(msg)
        logging.getLogger().error(msg)
        raise RuntimeError(msg) from exc
    val = objective_fun(params)
    val_f = float(val)
    if not np.isfinite(val_f):
        msg = f"JAXopt failed: non-finite final objective {val_f}"
        log.error(msg)
        logging.getLogger().error(msg)
        raise RuntimeError(msg)
    try:
        grad_norm = float(jnp.linalg.norm(jax.grad(objective_fun)(params)))
    except Exception:
        grad_norm = float("nan")
    log.info(
        "[Optimizer] Convergence status: iterations=%s final_objective=%.8g grad_norm=%.8g",
        getattr(state, "iter_num", "unknown"),
        val_f,
        grad_norm,
    )
    return np.asarray(params), state, val_f


def make_simple_objective(loss_data: Mapping, mode: DataMode, time_grid: Sequence[float], weights=None, defaults=None,
                          prior_weight=0.0, *, networkmodel_layout: bool = False, return_breakdown: bool = False,
                          y0=None, sys=None, slices=None, pinn_config=None, pinn_spec=None):
    """Create the scalar trajectory objective
    
    Args:
        loss_data: Input value used by this routine.
        mode: Input value used by this routine.
        time_grid: Input value used by this routine.
        weights: Input value used by this routine.
        defaults: Input value used by this routine.
        prior_weight: Input value used by this routine.
        networkmodel_layout: Input value used by this routine.
        return_breakdown: Input value used by this routine.
        y0: Input value used by this routine.
        sys: Input value used by this routine.
        slices: Input value used by this routine.
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
    """
    validate_loss_data(loss_data, mode)
    if pinn_config is not None and getattr(pinn_config, "enabled", False):
        from networkmodel.pinn.objective import make_pinn_objective

        return make_pinn_objective(
            loss_data=loss_data,
            mode=mode,
            time_grid=time_grid,
            weights=weights,
            defaults=defaults,
            prior_weight=prior_weight,
            networkmodel_layout=networkmodel_layout,
            return_breakdown=return_breakdown,
            y0=y0,
            sys=sys,
            slices=slices,
            pinn_config=pinn_config,
            pinn_spec=pinn_spec,
        )
    t = jnp.asarray(time_grid, dtype=jnp.float64)
    prot_map = np.asarray(loss_data["prot_map"])
    if len(prot_map):
        layout = str(loss_data.get("state_layout", "standard"))
        offsets_np = prot_map[:, 0]
        counts_np = prot_map[:, 1]
        if layout == "combinatorial":
            # each protein block: 1 mRNA + 2^n_sites states
            state_width = 1 + counts_np  # counts_np already holds 2^n for combinatorial
        else:
            # standard: 1 mRNA + 1 unphospho protein + n_sites phospho states
            state_width = 2 + counts_np
        state_dim = int(np.max(offsets_np + state_width))
    else:
        state_dim = 2
    y0 = jnp.ones(state_dim, dtype=jnp.float64) if y0 is None else jnp.asarray(y0, dtype=jnp.float64)
    if y0.shape[0] != state_dim:
        raise ValueError(
            f"Initial state dimension {y0.shape[0]} does not match prot_map-derived dimension {state_dim}.")
    model_rhs = make_networkmodel_rhs(sys, slices) if sys is not None and slices is not None else None
    defaults_j = _defaults_vector_jax(defaults, slices)

    def objective(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        if model_rhs is not None:
            physical_params = _unpack_theta_jax(theta, slices)
            rates = _flatten_params_for_slices_jax(physical_params, slices)
            Y = solve_diffrax(y0, t, params=theta, rhs=model_rhs)
        else:
            rates = jax.nn.softplus(theta)
            Y = solve_diffrax(y0, t, params=rates)
        if Y.shape[1] != state_dim:
            raise ValueError(f"Solved trajectory state width {Y.shape[1]} does not match expected {state_dim}.")
        total, breakdown = multimodal_loss_from_trajectory(
            Y, loss_data, mode, weights=weights, networkmodel_layout=networkmodel_layout
        )
        if defaults_j is not None and prior_weight:
            d = rates[: defaults_j.size] - defaults_j
            prior = float(prior_weight) * jnp.mean(d * d)
            total = total + prior
            breakdown["prior"] = prior
        if return_breakdown:
            breakdown["scalar_total"] = jnp.asarray(total, dtype=jnp.float64)
            return jnp.asarray(total, dtype=jnp.float64), breakdown
        return jnp.asarray(total, dtype=jnp.float64)

    return objective


def warn_deprecated_backend_options(options: Mapping | object | None, logger_obj=None):
    """Warn about accepted-but-ignored backend options
    
    Args:
        options: Input value used by this routine.
        logger_obj: Input value used by this routine.
    """
    if options is None:
        return
    log = logger_obj or logger
    get = options.get if isinstance(options, Mapping) else lambda k, d=None: getattr(options, k, d)
    optimizer = get("optimizer", get("solver", None))
    if optimizer and str(optimizer).lower() in {"pymoo", "optuna", "nsga3", "unsga3", "spea2", "de", "ga", "scipy"}:
        msg = f"[Solver] Deprecated solver option '{optimizer}' mapped to jaxopt.ProjectedGradient."
        log.warning(msg)
        logging.getLogger().warning(msg)
    for key in ("n_gen", "pop", "population_size", "use_custom_solver", "odeint", "solve_ivp"):
        val = get(key, None)
        if val is not None:
            msg = f"[Deprecated Config] {key}={val!r} is accepted but ignored by the JAXopt/Diffrax PhosKinTime path."
            log.warning(msg)
            logging.getLogger().warning(msg)
