"""JAX/JAXopt/Diffrax backend for PhosKinTime networkmodel and protwise fitting.

This module is intentionally independent from pandas and string identifiers inside
its differentiable functions. Data frames are converted to typed arrays before the
objective is called; JAX computations receive numeric arrays only.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
import diffrax

logger = logging.getLogger(__name__)

LAYERS = ("mrna", "protein", "phospho")
ALIASES = {"rna": "mrna", "mrna": "mrna", "protein": "protein", "prot": "protein", "phospho": "phospho",
           "pho": "phospho"}


@dataclass(frozen=True)
class DataMode:
    available_layers: tuple[str, ...]
    data_mode: str
    fit_mrna: bool
    fit_protein: bool
    fit_phospho: bool

    @property
    def active_loss_terms(self) -> tuple[str, ...]:
        return tuple(f"{layer}_loss" for layer in self.available_layers)

    @property
    def skipped_loss_terms(self) -> tuple[str, ...]:
        return tuple(f"{layer}_loss" for layer in LAYERS if layer not in self.available_layers)


def ensure_jax_float64() -> bool:
    jax.config.update("jax_enable_x64", True)
    return bool(jax.config.jax_enable_x64)


def detect_data_mode(*, mrna=None, protein=None, phospho=None, loss_data: Mapping | None = None,
                     logger_obj=None) -> DataMode:
    """Detect the non-empty mRNA/protein/phospho mode after loading input data."""

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
    solver_name: str = "Kvaerno4"
    rtol: float = 1e-5
    atol: float = 1e-7
    max_steps: int = 20000
    root_max_steps: int = 20

    def solver(self):
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
    rates = args
    n = y.shape[0]
    base = jnp.resize(rates, (n,))
    return base - (0.05 + jnp.abs(base)) * y



def _unpack_theta_jax(theta, slices):
    """Unpack raw optimizer theta into physical JAX arrays using params.py slice layout."""
    return {
        "A_i": jax.nn.softplus(theta[slices["A_i"]]),
        "B_i": jax.nn.softplus(theta[slices["B_i"]]),
        "C_i": jax.nn.softplus(theta[slices["C_i"]]),
        "D_i": jax.nn.softplus(theta[slices["D_i"]]),
        "E_i": jax.nn.softplus(theta[slices["E_i"]]),
        "c_k": jax.nn.softplus(theta[slices["c_k"]]),
        "tf_scale": jax.nn.softplus(theta[slices["tf_scale"]])[0],
        "Dp_i": jax.nn.softplus(theta[slices["Dp_i"]]),
    }


def make_networkmodel_rhs(sys, slices=None):
    """Build a JAX RHS matching the global networkmodel state layout.

    The RHS keeps all state reads anchored on idx.offset_y[i], so every protein
    reads its own mRNA/protein/phosphosite block instead of accidentally sharing
    state zero or the first phosphosite block.
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
    trans_from = jnp.asarray(getattr(sys, "trans_from", np.zeros(0, dtype=np.int32)), dtype=jnp.int32)
    trans_to = jnp.asarray(getattr(sys, "trans_to", np.zeros(0, dtype=np.int32)), dtype=jnp.int32)
    trans_site = jnp.asarray(getattr(sys, "trans_site", np.zeros(0, dtype=np.int32)), dtype=jnp.int32)
    trans_off = jnp.asarray(getattr(sys, "trans_off", np.zeros(N, dtype=np.int32)), dtype=jnp.int32)
    trans_n = jnp.asarray(getattr(sys, "trans_n", np.zeros(N, dtype=np.int32)), dtype=jnp.int32)
    max_trans = int(np.max(getattr(sys, "trans_n", np.zeros(N, dtype=np.int32)))) if N else 0

    def _params(args):
        if isinstance(args, dict):
            return args
        if slices is not None:
            return _unpack_theta_jax(jnp.asarray(args, dtype=jnp.float64), slices)
        c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale = args
        tf_scale = jnp.ravel(tf_scale)[0]
        return {"c_k": c_k, "A_i": A_i, "B_i": B_i, "C_i": C_i, "D_i": D_i, "Dp_i": Dp_i, "E_i": E_i, "tf_scale": tf_scale}

    def _synth(Ai, tf_scale, u_raw):
        u = u_raw / (1.0 + jnp.abs(u_raw))
        return jnp.where(
            u >= 0.0,
            Ai * (1.0 + (tf_scale * u) / (1.0 + u + 1e-6)),
            Ai / (1.0 + tf_scale * jnp.abs(u)),
        )

    def rhs(t, y, args):
        par = _params(args)
        K_raw = jax.vmap(lambda row: jnp.interp(t, kin_grid, row))(kin_Kmat)
        Kt = K_raw * par["c_k"]
        S_all = W @ Kt

        p_vals = []
        for i in range(N):
            off = offsets[i]
            drv = driver_map[i]
            if model_id == 2:
                ar = jnp.arange(max_states, dtype=jnp.int32)
                valid = ar < n_states[i]
                pos = jnp.minimum(off + 1 + ar, y.shape[0] - 1)
                vals = jnp.where(valid, y[pos], 0.0)
                total_p = jnp.sum(vals)
            else:
                ar = jnp.arange(max_sites, dtype=jnp.int32)
                valid = ar < n_sites[i]
                pos = jnp.minimum(off + 2 + ar, y.shape[0] - 1)
                vals = jnp.where(valid, y[pos], 0.0)
                total_p = y[off + 1] + jnp.sum(vals)
            p_vals.append(jnp.where(drv >= 0, Kt[jnp.maximum(drv, 0)], total_p))
        P_vec = jnp.stack(p_vals) if p_vals else jnp.asarray([], dtype=jnp.float64)
        TF_inputs = TF @ P_vec
        TF_inputs = (TF_inputs / tf_deg) / (1.0 + jnp.abs(TF_inputs / tf_deg))

        dy = jnp.zeros_like(y)
        for i in range(N):
            off = offsets[i]
            s_off = site_offsets[i]
            ns = n_sites[i]
            R = y[off]
            synth = _synth(par["A_i"][i], par["tf_scale"], TF_inputs[i])
            dy = dy.at[off].set(synth - par["B_i"][i] * R)
            if model_id == 2:
                # Combinatorial model: mirror models.combinatorial_rhs for the
                # Diffrax/JAX path. Translation feeds only mask 0, while explicit
                # phosphorylation transitions and implicit bit dephosphorylation
                # transitions move mass among all 2^n mask states.
                p0 = off + 1
                nst = n_states[i]
                if max_states > 0:
                    P0 = y[p0]
                    dy = dy.at[p0].add(par["C_i"][i] * R - par["D_i"][i] * P0)

                # Dephosphorylation and per-site phospho-state decay for masks m > 0.
                for m in range(1, max_states):
                    valid_m = m < nst
                    pos_m = jnp.minimum(p0 + m, y.shape[0] - 1)
                    Pm = jnp.where(valid_m, y[pos_m], 0.0)
                    dp_rate = 0.0
                    for j in range(max_sites):
                        bit_set = ((m >> j) & 1) != 0
                        valid_bit = valid_m & (j < ns) & bit_set
                        flat_j = jnp.minimum(s_off + j, par["Dp_i"].shape[0] - 1)
                        to = m ^ (1 << j)
                        pos_to = jnp.minimum(p0 + to, y.shape[0] - 1)
                        flux = jnp.where(valid_bit, par["E_i"][i] * Pm, 0.0)
                        dy = dy.at[pos_m].add(-flux)
                        dy = dy.at[pos_to].add(flux)
                        dp_rate = dp_rate + jnp.where(valid_bit, par["Dp_i"][flat_j] + par["D_i"][i], 0.0)
                    dy = dy.at[pos_m].add(-dp_rate * Pm)

                # Explicit phosphorylation transitions from the precomputed sparse
                # hypercube graph. Use S_all at the flattened site index so rates
                # remain differentiable through kinase scales and site parameters.
                for k in range(max_trans):
                    tr_idx = jnp.minimum(trans_off[i] + k, trans_from.shape[0] - 1)
                    valid_tr = k < trans_n[i]
                    frm = trans_from[tr_idx]
                    to = trans_to[tr_idx]
                    j = trans_site[tr_idx]
                    pos_frm = jnp.minimum(p0 + frm, y.shape[0] - 1)
                    pos_to = jnp.minimum(p0 + to, y.shape[0] - 1)
                    flat_j = jnp.minimum(s_off + j, S_all.shape[0] - 1)
                    flux = jnp.where(valid_tr, S_all[flat_j] * y[pos_frm], 0.0)
                    dy = dy.at[pos_frm].add(-flux)
                    dy = dy.at[pos_to].add(flux)
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
    """Solve an ODE with Diffrax Kvaerno4/Kvaerno5 and return (time, state)."""
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
    pm = jnp.asarray(prot_map, dtype=jnp.int32)
    return pm[:, 0], pm[:, 1]


def _safe_fold_change(values, base_values):
    return jnp.maximum(values, 1e-12) / jnp.maximum(base_values, 1e-12)


def _global_networkmodel_observable(Y, offsets, counts, n_sites, protein_idx, time_idx, *, layer, site_idx=None,
                                    base_idx=0, layout="standard", max_count=1):
    """Map global networkmodel state trajectories to fitted fold-change observables.

    Global state layout assumptions are deliberately kept in this networkmodel-only
    helper so protwise callers retain the legacy direct-state indexing path:
      * standard/sequential: [mRNA, unphosphorylated protein, phospho_site_0, ...]
      * combinatorial:       [mRNA, protein_state_0, ..., protein_state_(2^n-1)]
    Protein observations are total protein fold changes. Phospho observations are
    site fold changes: direct site states for standard layouts and bitwise sums of
    combinatorial protein states for MODEL==2 layouts.
    """
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
    weights = weights or {}
    prot_map = jnp.asarray(loss_data["prot_map"], dtype=jnp.int32)
    offsets, counts = _extract_offsets(prot_map)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    breakdown = {}
    layout = str(loss_data.get("state_layout", "standard"))
    n_sites = jnp.asarray(loss_data.get("n_sites", counts), dtype=jnp.int32)
    max_count = int(np.max(np.asarray(loss_data["prot_map"], dtype=np.int32)[:, 1])) if len(loss_data["prot_map"]) else 1

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
    x = jnp.asarray(x, dtype=jnp.float64)
    u = jnp.sort(x)[::-1]
    cssv = jnp.cumsum(u) - 1.0
    ind = jnp.arange(1, x.size + 1, dtype=jnp.float64)
    cond = u - cssv / ind > 0
    rho = jnp.sum(cond) - 1
    theta = cssv[rho] / (rho + 1.0)
    return jnp.maximum(x - theta, 0.0)


def project_alpha_blocks(alpha, block_ids):
    """Project alpha values onto one [0, 1] sum-to-one simplex per block."""
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
    """Project beta blocks to bounded affine sum-to-one sets without forcing non-negativity.

    Beta weights are allowed to be negative, so a standard non-negative simplex is
    biologically wrong. We first shift each block to satisfy the affine sum exactly,
    then clip to [-4, 4] and redistribute any residual sum error over entries that
    still have room. This keeps negative beta values when the optimum requires them.
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
    clipped = jnp.clip(jnp.asarray(theta, dtype=jnp.float64), jnp.asarray(lower, dtype=jnp.float64),
                       jnp.asarray(upper, dtype=jnp.float64))
    if fixed_mask is not None:
        clipped = jnp.where(jnp.asarray(fixed_mask, dtype=bool), jnp.asarray(fixed_values, dtype=jnp.float64), clipped)
    return clipped


@dataclass
class JaxoptResult:
    X: np.ndarray
    F: np.ndarray
    objective_value: float
    params: np.ndarray
    state: object
    data_mode: DataMode
    loss_breakdown: dict
    optimizer: str = "jaxopt.ProjectedGradient"


def optimize_scalar_objective(objective_fun, theta0, lower, upper, *, maxiter=20000, tol=1e-6, fixed_mask=None,
                              fixed_values=None, logger_obj=None):
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

    solver = jaxopt.ProjectedGradient(fun=objective_fun, projection=projection, maxiter=int(maxiter), tol=float(tol),
                                      verbose=1)
    init = project_bounds(theta0, lower_j, upper_j, fixed_mask_j, fixed_values_j)
    params, state = solver.run(init, hyperparams_proj=(lower_j, upper_j))
    val = objective_fun(params)
    val_f = float(val)
    if not np.isfinite(val_f):
        log.error("[Optimizer] JAXopt failed: final scalar objective is not finite (%s).", val_f)
        raise RuntimeError("JAXopt optimization failed: final scalar objective is not finite.")
    try:
        grad_norm = float(jnp.linalg.norm(jax.grad(objective_fun)(params)))
    except Exception:  # diagnostic only; keep optimization result usable if grad logging fails
        grad_norm = float("nan")
    log.info(
        "[Optimizer] Convergence status: iterations=%s final scalar objective=%.8g grad_norm=%.8g",
        getattr(state, "iter_num", "unknown"),
        val_f,
        grad_norm,
    )
    return np.asarray(params), state, val_f


def make_simple_objective(loss_data: Mapping, mode: DataMode, time_grid: Sequence[float], weights=None, defaults=None,
                          prior_weight=0.0, *, networkmodel_layout: bool = False, return_breakdown: bool = False,
                          y0=None, sys=None, slices=None):
    validate_loss_data(loss_data, mode)
    t = jnp.asarray(time_grid, dtype=jnp.float64)
    prot_map = np.asarray(loss_data["prot_map"])
    if len(prot_map):
        layout = str(loss_data.get("state_layout", "standard"))
        state_width = prot_map[:, 1] + (1 if layout == "combinatorial" else 2)
        state_dim = int(np.max(prot_map[:, 0] + np.maximum(state_width, 2)))
    else:
        state_dim = 2
    y0 = jnp.ones(state_dim, dtype=jnp.float64) if y0 is None else jnp.asarray(y0, dtype=jnp.float64)
    if y0.shape[0] != state_dim:
        raise ValueError(f"Initial state dimension {y0.shape[0]} does not match prot_map-derived dimension {state_dim}.")
    model_rhs = make_networkmodel_rhs(sys, slices) if sys is not None and slices is not None else None
    defaults_j = None if defaults is None else jnp.asarray(defaults, dtype=jnp.float64)

    def objective(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        if model_rhs is not None:
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
    if options is None:
        return
    log = logger_obj or logger
    get = options.get if isinstance(options, Mapping) else lambda k, d=None: getattr(options, k, d)
    optimizer = get("optimizer", get("solver", None))
    if optimizer and str(optimizer).lower() in {"pymoo", "optuna", "nsga3", "unsga3", "spea2", "de", "ga", "scipy"}:
        log.warning(
            "[Deprecated Config] optimizer/solver=%r is accepted for compatibility and mapped to jaxopt.ProjectedGradient.",
            optimizer)
    for key in ("n_gen", "pop", "population_size", "use_custom_solver", "odeint", "solve_ivp"):
        val = get(key, None)
        if val is not None:
            log.warning("[Deprecated Config] %s=%r is accepted but ignored by the JAXopt/Diffrax PhosKinTime path.",
                        key, val)
