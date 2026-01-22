from typing import Tuple, Callable

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from global_model.config import ODE_MAX_STEPS, ODE_ABS_TOL, ODE_REL_TOL
from global_model.lossfn import LOSS_FN
from global_model.params import unpack_params
from global_model.simulate import simulate_odeint


class GlobalODE_MOO(ElementwiseProblem):
    """
    Elementwise multiobjective optimization problem.
    Objectives: [Protein Fit, RNA Fit, Phospho Fit]
    Each objective includes normalized MSE and a prior regularization term.
    """

    def __init__(self, sys, slices, loss_data, defaults, lambdas, time_grid,
                 xl, xu, fail_value=1e12, elementwise_runner=None):

        # We optimize for 3 objectives: Protein, RNA, and Phospho
        super().__init__(
            n_var=len(xl),
            n_obj=3,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            elementwise_runner=elementwise_runner
        )
        self.sys = sys
        self.slices = slices
        self.loss_data = loss_data
        self.defaults = defaults
        self.lambdas = lambdas
        self.time_grid = time_grid
        self.fail_value = float(fail_value)

        # --- MODALITY NORMALIZATION FACTORS ---
        # Normalize by the sum of weights (number of data points * weight boost)
        # This ensures that an MSE of 0.1 in RNA is 'weighted' the same as 0.1 in Phospho
        self.norm_p = 1.0 / max(1e-6, np.sum(loss_data["w_prot"]))
        self.norm_r = 1.0 / max(1e-6, np.sum(loss_data["w_rna"]))
        self.norm_ph = 1.0 / max(1e-6, np.sum(loss_data["w_pho"]))

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Parameter Unpacking
        p = unpack_params(x, self.slices)
        self.sys.update(**p)

        # 2. Prior Regularization (Adherence to kinopt/tfopt priors)
        # Calculate mean squared percentage error from defaults
        reg_accumulator = 0.0
        reg_count = 0
        for k in ["A_i", "B_i", "C_i", "D_i", "E_i"]:
            # Relative difference prevents large basal rates from dominating the penalty
            diff = (p[k] - self.defaults[k]) / (self.defaults[k] + 1e-6)
            reg_accumulator += np.sum(diff ** 2)
            reg_count += diff.size

        # Global prior penalty scaled by lambda_prior
        prior_penalty = self.lambdas["prior"] * (reg_accumulator / max(1, reg_count))

        # 3. Simulation
        try:
            Y = simulate_odeint(
                self.sys,
                self.time_grid,
                rtol=ODE_REL_TOL,
                atol=ODE_ABS_TOL,
                mxstep=ODE_MAX_STEPS
            )
        except Exception:
            out["F"] = np.full(3, self.fail_value)
            return

        if Y is None or not np.all(np.isfinite(Y)):
            out["F"] = np.full(3, self.fail_value)
            return

        # 4. Loss Calculation (Raw Sums)
        loss_p_sum, loss_r_sum, loss_ph_sum = LOSS_FN(
            np.ascontiguousarray(Y),
            self.loss_data["p_prot"], self.loss_data["t_prot"], self.loss_data["obs_prot"], self.loss_data["w_prot"],
            self.loss_data["p_rna"], self.loss_data["t_rna"], self.loss_data["obs_rna"], self.loss_data["w_rna"],
            self.loss_data["p_pho"], self.loss_data["s_pho"], self.loss_data["t_pho"], self.loss_data["obs_pho"],
            self.loss_data["w_pho"],
            self.loss_data["prot_map"],
            self.loss_data["prot_base_idx"], self.loss_data["rna_base_idx"], self.loss_data["pho_base_idx"]
        )

        # 5. Normalization and Weighted Objectives
        # We apply: (Raw Sum / Total Weights) * User Lambda
        obj_protein = (loss_p_sum * self.norm_p) * self.lambdas["protein"]
        obj_rna = (loss_r_sum * self.norm_r) * self.lambdas["rna"]
        obj_phospho = (loss_ph_sum * self.norm_ph) * self.lambdas["phospho"]

        # 6. Assemble Final Objectives
        # Distribute the prior penalty across all objectives to guide the Pareto front
        # towards biological plausibility in every modality.
        out["F"] = np.array([
            obj_protein + prior_penalty,
            obj_rna + prior_penalty,
            obj_phospho + prior_penalty
        ], dtype=float)


def get_weight_options(
        time_points,
        *,
        rna_time_points=None,
        early_window=None,
        center=None,
        baseline=None,
        eps=1e-12,
):
    """
    Return many weighting schemes as callables.

    Each scheme is a function f(t)->w (vectorized). You can apply:
        df["w"] = df["time"].map(lambda t: wmap[t])  (or use vectorized form)

    Parameters
    ----------
    time_points : array-like
        All times you want schemes to support (e.g. np.unique(concat(TIME_POINTS, TIME_POINTS_RNA))).
    rna_time_points : array-like or None
        Optional; used only for a couple RNA-friendly schemes.
    early_window : float or None
        Times <= early_window are "early". If None, uses ~20% quantile of time_points.
    center : float or None
        Center for gaussian/logistic. If None, uses median(time_points).
    baseline : float or None
        Baseline time (for "from_baseline" schemes). If None, uses min(time_points).
    eps : float
        Numerical floor.

    Returns
    -------
    dict[str, callable]
        name -> function f(t)->w
    """
    t = np.asarray(time_points, dtype=float)
    tmin, tmax = float(np.min(t)), float(np.max(t))
    trng = max(tmax - tmin, eps)
    tn = (t - tmin) / trng  # normalized to [0,1]

    if early_window is None:
        early_window = float(np.quantile(t, 0.20))
    if center is None:
        center = float(np.median(t))
    if baseline is None:
        baseline = tmin

    def _clip_pos(x):
        return np.maximum(np.asarray(x, dtype=float), eps)

    def _normalize_mean1(w):
        w = np.asarray(w, dtype=float)
        m = float(np.mean(w)) if w.size else 1.0
        return w / max(m, eps)

    # Precompute some scalars
    c = (center - tmin) / trng
    sigma = 0.18  # width on normalized axis; tweak if needed
    k = 10.0  # logistic sharpness
    ewin = (early_window - tmin) / trng

    schemes = {}

    # 1) Uniform
    schemes["uniform"] = lambda tt: np.ones_like(np.asarray(tt, dtype=float))

    # 2) Linear early emphasis (your current style): higher weight for smaller t
    schemes["linear_early"] = lambda tt: 1.0 + (tmax - np.asarray(tt, float)) / max(tmax, eps)

    # 3) Linear late emphasis
    schemes["linear_late"] = lambda tt: 1.0 + (np.asarray(tt, float) - tmin) / max(trng, eps)

    # 4) Quadratic early emphasis
    schemes["quad_early"] = lambda tt: 1.0 + ((tmax - np.asarray(tt, float)) / max(trng, eps)) ** 2

    # 5) Quadratic late emphasis
    schemes["quad_late"] = lambda tt: 1.0 + ((np.asarray(tt, float) - tmin) / max(trng, eps)) ** 2

    # 6) Exponential early emphasis
    schemes["exp_early"] = lambda tt: np.exp(2.0 * (1.0 - (np.asarray(tt, float) - tmin) / max(trng, eps)))

    # 7) Exponential late emphasis
    schemes["exp_late"] = lambda tt: np.exp(2.0 * ((np.asarray(tt, float) - tmin) / max(trng, eps)))

    # 8) Inverse time (huge early); safe with eps
    schemes["inv_time"] = lambda tt: 1.0 / _clip_pos(np.asarray(tt, float) - tmin + 1.0)

    # 9) Inverse sqrt time (milder early)
    schemes["inv_sqrt_time"] = lambda tt: 1.0 / np.sqrt(_clip_pos(np.asarray(tt, float) - tmin + 1.0))

    # 10) Log early emphasis (mild)
    schemes["log_early"] = lambda tt: 1.0 + np.log1p((tmax - np.asarray(tt, float)) / max(trng, eps))

    # 11) Piecewise: upweight early window only
    schemes["piecewise_early_boost"] = lambda tt, boost=4.0: np.where(
        ((np.asarray(tt, float) - tmin) / max(trng, eps)) <= ewin,
        boost,
        1.0
    )

    # 12) Gaussian around a center time (focus mid window)
    schemes["gaussian_center"] = lambda tt: 1.0 + np.exp(
        -0.5 * (((((np.asarray(tt, float) - tmin) / trng) - c) / sigma) ** 2))

    # 13) Logistic early (smooth step that decays with time)
    # weight ~2 at early, ~1 at late
    schemes["logistic_early"] = lambda tt: 1.0 + 1.0 / (1.0 + np.exp(k * (((np.asarray(tt, float) - tmin) / trng) - c)))

    # 14) Baseline-anchored: emphasize far from baseline (useful if baseline is special like RNA at 4.0)
    schemes["distance_from_baseline"] = lambda tt: 1.0 + np.abs(np.asarray(tt, float) - float(baseline)) / max(trng,
                                                                                                               eps)

    # Optional: a scheme that is RNA-friendly if you pass rna_time_points
    if rna_time_points is not None:
        rna_tp = np.asarray(rna_time_points, dtype=float)
        rna_set = set(np.round(rna_tp, 12).tolist())
        # 15) Boost weights only on RNA measurement times
        schemes["boost_rna_times"] = lambda tt: np.where(
            np.isin(np.round(np.asarray(tt, float), 12), list(rna_set)),
            2.0,
            1.0
        )

    # Return both raw and mean-normalized variants (so magnitude doesn’t change step size too much)
    out = {}
    for name, f in schemes.items():
        out[name] = f
        out[name + "_mean1"] = lambda tt, ff=f: _normalize_mean1(ff(tt))

    return out


def build_weight_functions(
        time_points_protein: np.ndarray,
        time_points_rna: np.ndarray,
        scheme_prot_pho: str = "uniform",
        scheme_rna: str = "uniform",
        early_window_prot_pho: float = 2.0,
        early_window_rna: float = 15.0,
) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Build modality-specific weight functions from selected schemes.

    Parameters
    ----------
    time_points_protein:
        Timepoints for protein/phospho modality (your TIME_POINTS_PROTEIN).
    time_points_rna:
        Timepoints for RNA modality (your TIME_POINTS_RNA).
    scheme_prot_pho:
        Weighting scheme name for protein/phospho (must exist in get_weight_options output dict).
    scheme_rna:
        Weighting scheme name for RNA (must exist in get_weight_options output dict).
    early_window_prot_pho:
        Early window (minutes) used when building schemes for protein/phospho.
    early_window_rna:
        Early window (minutes) used when building schemes for RNA.

    Returns
    -------
    w_prot_pho, w_rna:
        Two callables that take np.ndarray of times and return np.ndarray weights.
    """
    tp_prot_pho = np.asarray(time_points_protein, dtype=float)
    tp_rna = np.asarray(time_points_rna, dtype=float)

    schemes_prot_pho = get_weight_options(tp_prot_pho, early_window=early_window_prot_pho)
    schemes_rna = get_weight_options(tp_rna, early_window=early_window_rna)

    if scheme_prot_pho not in schemes_prot_pho:
        raise KeyError(
            f"Unknown protein/phospho weighting scheme '{scheme_prot_pho}'. "
            f"Available: {sorted(schemes_prot_pho.keys())}"
        )
    if scheme_rna not in schemes_rna:
        raise KeyError(
            f"Unknown RNA weighting scheme '{scheme_rna}'. "
            f"Available: {sorted(schemes_rna.keys())}"
        )

    w_prot_pho = lambda tt: schemes_prot_pho[scheme_prot_pho](tt)
    w_rna = lambda tt: schemes_rna[scheme_rna](tt)

    return w_prot_pho, w_rna
