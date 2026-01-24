"""
Multi-Objective Optimization Problem Definition.

This module defines the `GlobalODE_MOO` class, which wraps the biological simulation
and loss calculation into a format compatible with the `pymoo` optimization framework.

**Key Concepts:**
1.  **Elementwise Problem:** Each member of the population (a parameter set) is evaluated individually.
2.  **Three Objectives:** The optimizer simultaneously minimizes error for:
    * Protein Abundance
    * mRNA Abundance
    * Phosphorylation Levels
3.  **Prior Regularization:** A penalty term ensures parameters stay close to biologically plausible
    priors derived from upstream analysis (KinOpt/TFOpt). This penalty is added to *all* objectives
    to steer the entire Pareto front toward realistic regions.


"""

from typing import Tuple, Callable

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from global_model.config import ODE_MAX_STEPS, ODE_ABS_TOL, ODE_REL_TOL
from global_model.lossfn import LOSS_FN
from global_model.params import unpack_params
from global_model.simulate import simulate_odeint


class GlobalODE_MOO(ElementwiseProblem):
    """
    Defines the multi-objective optimization problem for the ODE model.

    Attributes:
        sys (System): The biological system object (contains topology and matrices).
        slices (dict): Mapping of parameter names to indices in the decision vector `x`.
        loss_data (dict): Pre-processed experimental data arrays for fast loss calculation.
        defaults (dict): Prior parameter values for regularization.
        lambdas (dict): Hyperparameters controlling the strength of penalties/weights.
        time_grid (np.ndarray): The common time grid for simulation.
        fail_value (float): Penalty value assigned to objectives if simulation fails.
    """

    def __init__(self, sys, slices, loss_data, defaults, lambdas, time_grid,
                 xl, xu, fail_value=1e12, elementwise_runner=None):
        """
        Initialize the MOO problem.

        Args:
            sys: System object.
            slices: Parameter slice dictionary.
            loss_data: Dictionary of contiguous arrays for loss function.
            defaults: Dictionary of default/prior parameter values.
            lambdas: Dictionary containing weights for 'protein', 'rna', 'phospho', and 'prior'.
            time_grid: Simulation timepoints.
            xl: Lower bounds for parameters.
            xu: Upper bounds for parameters.
            fail_value: Objective value to return if ODE solver diverges.
            elementwise_runner: Optional runner for parallel execution.
        """
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
        # despite potentially vastly different data counts or scales.
        self.norm_p = 1.0 / max(1e-6, np.sum(loss_data["w_prot"]))
        self.norm_r = 1.0 / max(1e-6, np.sum(loss_data["w_rna"]))
        self.norm_ph = 1.0 / max(1e-6, np.sum(loss_data["w_pho"]))

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates a single parameter set `x`.

        Steps:
        1.  Unpack `x` into model parameters (A_i, c_k, etc.).
        2.  Calculate regularization penalty (deviation from priors).
        3.  Simulate the ODE system.
        4.  Compute loss vs. experimental data.
        5.  Return the 3 objective values.
        """
        # 1. Parameter Unpacking
        p = unpack_params(x, self.slices)
        self.sys.update(**p)

        # 2. Prior Regularization (Adherence to kinopt/tfopt priors)
        # Calculate mean squared percentage error from defaults.
        # This keeps the parameters biologically grounded.
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
            # If solver crashes, return high penalty
            out["F"] = np.full(3, self.fail_value)
            return

        # Check for NaNs or Inf in result
        if Y is None or not np.all(np.isfinite(Y)):
            out["F"] = np.full(3, self.fail_value)
            return

        # 4. Loss Calculation (Raw Sums)
        # Uses the JIT-compiled loss function for speed
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
    Generates a library of time-dependent weighting schemes.



    Time-series data often requires non-uniform weighting. For example:
    - **Early Transients:** Often contain the most kinetic information, so we might boost early timepoints.
    - **Late Steady-State:** Might be noisy or less relevant for initial kinetics.

    Parameters
    ----------
    time_points : array-like
        All times you want schemes to support.
    rna_time_points : array-like or None
        Optional; used only for a couple RNA-friendly schemes.
    early_window : float or None
        Times <= early_window are "early". If None, uses ~20% quantile.
    center : float or None
        Center for gaussian/logistic. If None, uses median.
    baseline : float or None
        Baseline time (for "from_baseline" schemes).
    eps : float
        Numerical floor to prevent division by zero.

    Returns
    -------
    dict[str, callable]
        Dictionary mapping scheme names (e.g., 'linear_early') to functions `f(t) -> weight`.
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

    # 1) Uniform: Equal weight for all t
    schemes["uniform"] = lambda tt: np.ones_like(np.asarray(tt, dtype=float))

    # 2) Linear early emphasis: Higher weight at t=0, decays linearly
    schemes["linear_early"] = lambda tt: 1.0 + (tmax - np.asarray(tt, float)) / max(tmax, eps)

    # 3) Linear late emphasis: Higher weight at t=max
    schemes["linear_late"] = lambda tt: 1.0 + (np.asarray(tt, float) - tmin) / max(trng, eps)

    # 4) Quadratic early emphasis: Decays as t^2
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

    # 11) Piecewise: Step function boosting early window only
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
    Factory to build modality-specific weight functions based on configuration.

    Parameters
    ----------
    time_points_protein:
        Timepoints for protein/phospho modality.
    time_points_rna:
        Timepoints for RNA modality.
    scheme_prot_pho:
        Weighting scheme name for protein/phospho (e.g., 'linear_early').
    scheme_rna:
        Weighting scheme name for RNA (e.g., 'uniform').
    early_window_prot_pho:
        Definition of 'early' (in minutes) for protein data.
    early_window_rna:
        Definition of 'early' (in minutes) for RNA data.

    Returns
    -------
    w_prot_pho, w_rna:
        Two callables that take time arrays and return weight arrays.
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