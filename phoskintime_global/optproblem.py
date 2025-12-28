import numpy as np
from pymoo.core.problem import ElementwiseProblem

from phoskintime_global.config import ODE_MAX_STEPS, ODE_ABS_TOL
from phoskintime_global.lossfn import LOSS_FN
from phoskintime_global.params import unpack_params
from phoskintime_global.simulate import simulate_odeint


class GlobalODE_MOO(ElementwiseProblem):
    """
    Elementwise multiobjective:
      F = [prot_mse, rna_mse, reg_loss]
    """

    def __init__(self, sys, slices, loss_data, defaults, lambdas, time_grid,
                 xl, xu, fail_value=1e12, elementwise_runner=None):
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

    def _evaluate(self, x, out, *args, **kwargs):
        # 1) unpack + update
        p = unpack_params(x, self.slices)
        self.sys.update(**p)

        # 2) reg term (same as before, but keep as separate objective)
        reg = 0.0
        count = 0
        for k in ["A_i", "B_i", "C_i", "D_i", "E_i"]:
            diff = (p[k] - self.defaults[k]) / (self.defaults[k] + 1e-6)
            reg += float(np.sum(diff * diff))
            count += diff.size
        reg_loss = self.lambdas["prior"] * (reg / max(1, count))

        # 3) simulate (odeint + njit RHS + njit Jacobian)
        try:
            Y = simulate_odeint(
                self.sys,
                self.time_grid,
                rtol=ODE_ABS_TOL,
                atol=ODE_ABS_TOL,
                mxstep=ODE_MAX_STEPS
            )
        except Exception:
            out["F"] = np.array([self.fail_value, self.fail_value, self.fail_value], dtype=float)
            return

        if Y is None or Y.size == 0 or not np.all(np.isfinite(Y)):
            out["F"] = np.array([self.fail_value, self.fail_value, self.fail_value], dtype=float)
            return

        # Ensure (T, state_dim) contiguous
        Y = np.ascontiguousarray(Y)
        loss_p_sum, loss_r_sum = LOSS_FN(
            Y,
            self.loss_data["p_prot"], self.loss_data["t_prot"], self.loss_data["obs_prot"], self.loss_data["w_prot"],
            self.loss_data["p_rna"], self.loss_data["t_rna"], self.loss_data["obs_rna"], self.loss_data["w_rna"],
            self.loss_data["prot_map"], self.loss_data["rna_base_idx"]
        )

        prot_mse = loss_p_sum / self.loss_data["n_p"]
        rna_mse = loss_r_sum / self.loss_data["n_r"]

        out["F"] = np.array([prot_mse, rna_mse, reg_loss], dtype=float)

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
    k = 10.0      # logistic sharpness
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
    schemes["piecewise_early_boost"] = lambda tt: np.where(
        ((np.asarray(tt, float) - tmin) / max(trng, eps)) <= ewin,
        3.0,  # early boost factor
        1.0
    )

    # 12) Gaussian around a center time (focus mid window)
    schemes["gaussian_center"] = lambda tt: 1.0 + np.exp(-0.5 * (((((np.asarray(tt, float) - tmin) / trng) - c) / sigma) ** 2))

    # 13) Logistic early (smooth step that decays with time)
    # weight ~2 at early, ~1 at late
    schemes["logistic_early"] = lambda tt: 1.0 + 1.0 / (1.0 + np.exp(k * (((np.asarray(tt, float) - tmin) / trng) - c)))

    # 14) Baseline-anchored: emphasize far from baseline (useful if baseline is special like RNA at 4.0)
    schemes["distance_from_baseline"] = lambda tt: 1.0 + np.abs(np.asarray(tt, float) - float(baseline)) / max(trng, eps)

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
