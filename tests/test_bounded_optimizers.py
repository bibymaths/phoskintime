"""Bounded optimizer protocol (KNOWN_ISSUES O01): networkmodel.optimizers.minimize_bounded.

Both optimizers (jaxopt ProjectedGradient, the default, and scipy L-BFGS-B) on
- bounded quadratics with the optimum inside and at a bound (KKT residual ~ 0, the correct active set);
- a fixed coordinate (stays exactly at its fixed value);
- synthetic recovery of known rates of a small mechanistic ODE from noisy data;
plus: the projected_gradient arm returns exactly what backend.optimize_scalar_objective returns, and a jax-gradient
vs central-finite-difference check (step sweep) at an interior point of the B4 networkmodel objective (skipped when the
B4 input data are not present: they are not in the repository).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OPTS = ("projected_gradient", "lbfgsb")


def _quad(c, w):
    import jax.numpy as jnp

    c = jnp.asarray(c, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    return lambda x: jnp.sum(w * (x - c) ** 2)


@pytest.mark.parametrize("opt", OPTS)
def test_quadratic_optimum_inside(opt):
    from networkmodel.optimizers import minimize_bounded

    c = np.array([0.3, -0.2, 0.7])
    r = minimize_bounded(_quad(c, [1.0, 2.0, 0.5]), np.zeros(3), -np.ones(3), np.ones(3), optimizer=opt,
                         maxiter=5000, tol=1e-10)
    assert r.converged, r.stop_reason
    np.testing.assert_allclose(r.theta, c, atol=1e-6)
    assert r.kkt_residual < 1e-6
    assert r.active_lower == [] and r.active_upper == []
    assert r.optimizer == opt and r.criteria["optimizer"] == opt


@pytest.mark.parametrize("opt", OPTS)
def test_quadratic_optimum_at_bounds(opt):
    from networkmodel.optimizers import minimize_bounded

    c = np.array([2.0, 0.1, -3.0, 0.4])  # coordinate 0 above its upper bound, 2 below its lower bound
    lo, hi = np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    r = minimize_bounded(_quad(c, np.ones(4)), np.zeros(4), lo, hi, optimizer=opt, maxiter=5000, tol=1e-10)
    assert r.converged, r.stop_reason
    assert r.theta[0] == hi[0] and r.theta[2] == lo[2]
    np.testing.assert_allclose(r.theta[[1, 3]], c[[1, 3]], atol=1e-6)
    assert r.active_upper == [0] and r.active_lower == [2]
    assert r.kkt_residual < 1e-6  # the gradient at an active bound points outward: the projected step is 0


@pytest.mark.parametrize("opt", OPTS)
def test_fixed_coordinate_stays_exactly_fixed(opt):
    from networkmodel.optimizers import minimize_bounded

    c = np.array([0.3, -0.2, 0.7])
    mask = np.array([False, True, False])
    fixed = np.array([0.0, 0.37, 0.0])
    r = minimize_bounded(_quad(c, np.ones(3)), np.zeros(3), -np.ones(3), np.ones(3), fixed_mask=mask,
                         fixed_values=fixed, optimizer=opt, maxiter=5000, tol=1e-10)
    assert r.theta[1] == 0.37  # exact, not approximately
    np.testing.assert_allclose(r.theta[[0, 2]], c[[0, 2]], atol=1e-6)
    assert 1 not in r.active_lower + r.active_upper


def test_projected_gradient_arm_is_the_existing_optimizer():
    import jax.numpy as jnp

    from networkmodel.backend import optimize_scalar_objective
    from networkmodel.optimizers import minimize_bounded

    f = lambda x: jnp.sum((x - jnp.array([0.2, 0.9, -0.4])) ** 4) + jnp.sum(jnp.sin(x))  # noqa: E731
    args = (np.array([0.5, 0.5, 0.5]), -np.ones(3), np.ones(3))
    p_old, s_old, v_old = optimize_scalar_objective(f, *args, maxiter=37, tol=1e-6)
    r = minimize_bounded(f, *args, optimizer="projected_gradient", maxiter=37, tol=1e-6)
    assert np.array_equal(r.theta, np.asarray(p_old)) and r.final_objective == v_old
    assert r.iterations == int(s_old.iter_num)
    assert r.n_fun_evals is None and r.n_grad_evals is None  # jaxopt does not expose counts


def test_unknown_optimizer_rejected():
    from networkmodel.optimizers import minimize_bounded

    with pytest.raises(ValueError, match="optimizer must be one of"):
        minimize_bounded(lambda x: (x ** 2).sum(), np.zeros(1), -np.ones(1), np.ones(1), optimizer="adam")


@pytest.mark.parametrize("opt", OPTS)
def test_synthetic_mechanistic_recovery(opt):
    """x' = -k1 x, y' = k1 x - k2 y; k = softplus(raw) as in networkmodel; noisy observations of x and y."""
    import jax.numpy as jnp

    from networkmodel.backend import DiffraxSolverConfig, solve_diffrax
    from networkmodel.optimizers import minimize_bounded

    k_true = np.array([0.30, 0.08])
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0])
    y0 = jnp.array([1.0, 0.0])
    cfg = DiffraxSolverConfig(rtol=1e-10, atol=1e-10)

    def rhs(tt, y, k):
        return jnp.array([-k[0] * y[0], k[0] * y[0] - k[1] * y[1]])

    clean = np.asarray(solve_diffrax(y0, jnp.asarray(t), params=jnp.asarray(k_true), rhs=rhs, config=cfg))
    rng = np.random.default_rng(20261127)
    data = jnp.asarray(clean + 0.005 * rng.standard_normal(clean.shape))

    def objective(raw):
        k = jnp.logaddexp(0.0, raw)
        return jnp.mean((solve_diffrax(y0, jnp.asarray(t), params=k, rhs=rhs, config=cfg) - data) ** 2)

    inv = lambda p: np.log(np.expm1(p))  # noqa: E731
    lo, hi = inv(np.array([1e-3, 1e-3])), inv(np.array([5.0, 5.0]))
    r = minimize_bounded(objective, inv(np.array([1.0, 1.0])), lo, hi, optimizer=opt, maxiter=3000, tol=1e-9)
    k_hat = np.logaddexp(0.0, r.theta)
    np.testing.assert_allclose(k_hat, k_true, rtol=0.05)
    assert r.kkt_residual < 1e-5
    assert r.final_objective < 2 * 0.005 ** 2


_B4_INPUTS = [ROOT / "data" / "input1.csv", ROOT / "data" / "input3.csv",
              ROOT / "results_node_subnetworks" / "ABL2" / "ABL2_input2.csv"]

_FD_SCRIPT = r"""
import json, multiprocessing as mp, sys
import numpy as np
mp.set_start_method("fork", force=True)
out_dir = sys.argv[1]
sys.argv = ["networkmodel.runner", "--output-dir", out_dir]
import networkmodel.OptimizationProblem as OP
box = {}
class _Stop(Exception):
    pass
def _capture(self, theta0, maxiter=50, tol=1e-6):
    box["p"] = self
    raise _Stop()
OP.GlobalODEScalarObjective.solve = _capture
import networkmodel.runner as R
try:
    R.main()
except _Stop:
    pass
import jax
import jax.numpy as jnp
p = box["p"]
lo, hi = p.xl, p.xu
theta = lo + np.linspace(0.2, 0.8, lo.size) * (hi - lo)   # an interior point of the B4 box
f = jax.jit(p.objective)
g = np.asarray(jax.jit(jax.grad(p.objective))(jnp.asarray(theta)))
steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
rows = []
for i in range(theta.size):
    errs = []
    for h in steps:
        e = np.zeros_like(theta); e[i] = h
        fd = (float(f(jnp.asarray(theta + e))) - float(f(jnp.asarray(theta - e)))) / (2 * h)
        errs.append(abs(fd - g[i]) / max(abs(g[i]), 1e-8))
    rows.append({"i": i, "grad": float(g[i]), "rel_err": errs})
print("FDJSON" + json.dumps({"steps": steps, "rows": rows}))
"""


@pytest.mark.skipif(not all(p.exists() for p in _B4_INPUTS), reason="B4 input data not present (not in the repo)")
def test_jax_gradient_matches_central_fd_at_interior_point_of_b4(tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "-c", _FD_SCRIPT, str(tmp_path / "out")], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("FDJSON")), None)
    assert line, proc.stdout[-3000:] + proc.stderr[-3000:]
    res = json.loads(line[len("FDJSON"):])
    for row in res["rows"]:
        best = min(row["rel_err"])
        # central FD is O(h^2) and limited by the ODE solver tolerance (1e-8): the best step must agree to 1e-4
        assert best < 1e-4 or abs(row["grad"]) < 1e-10, row


def test_lbfgsb_ftol_stop_is_converged_only_at_a_first_order_point():
    """O01 review M1: an ftol stop with KKT residual > pgtol must not be labelled converged."""
    from networkmodel.optimizers import minimize_bounded

    c = np.array([0.3, -0.2])
    r = minimize_bounded(_quad(c, [1e6, 1e-6]), np.array([0.9, 0.9]), -np.ones(2), np.ones(2), optimizer="lbfgsb",
                         lbfgsb_options={"ftol": 1e-3, "pgtol": 1e-12})
    assert r.converged == (r.kkt_residual <= 1e-12)
    assert "KKT residual" in r.stop_reason


def test_lbfgsb_nonfinite_gradient_is_counted_and_not_converged():
    import jax
    import jax.numpy as jnp

    from networkmodel.optimizers import minimize_bounded

    @jax.custom_vjp
    def f(x):
        return jnp.sum((x - 0.2) ** 2)

    def f_fwd(x):
        return f(x), x

    def f_bwd(x, ct):
        return (jnp.where(x[0] < 0.5, jnp.nan, 2 * (x - 0.2)) * ct,)

    f.defvjp(f_fwd, f_bwd)
    r = minimize_bounded(f, np.array([0.9, 0.9]), -np.ones(2), np.ones(2), optimizer="lbfgsb", diagnostics=False)
    assert not r.converged
    assert "non-finite gradient" in r.stop_reason


def test_value_and_grad_cache_is_shared_by_bound_methods():
    """O01 review R2: problem.objective is a new bound-method object per access; one compiled function per problem."""
    import jax.numpy as jnp

    from networkmodel import optimizers

    class P:
        def objective(self, x):
            return jnp.sum(x ** 2)

    p = P()
    n0 = len(optimizers._VG_CACHE)
    a = optimizers._value_and_grad(p.objective)
    b = optimizers._value_and_grad(p.objective)
    assert a is b and len(optimizers._VG_CACHE) == n0 + 1


def test_kkt_diagnostics_degenerate_box_has_zero_projected_gradient():
    from networkmodel.optimizers import kkt_diagnostics

    d = kkt_diagnostics(None, np.array([0.5, 0.0]), np.array([0.5, -1.0]), np.array([0.5, 1.0]),
                        grad=np.array([3.0, 0.0]))
    assert d["proj_grad_norm"] == 0.0 and d["kkt_residual"] == 0.0
