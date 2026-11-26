"""P1: protwise RNA predictions must be read at the RNA observation times (config [timepoints] rna),
not at the last n protein times. One code path for fit (objective) and export (solve_protwise_ode).
"""
import numpy as np

MODEL = "distmod"
NUM_PSITES = 2


def _setup():
    from config.constants import TIME_POINTS, TIME_POINTS_RNA, get_num_params

    rng = np.random.default_rng(11)
    params = rng.uniform(0.05, 0.8, get_num_params(MODEL, NUM_PSITES))
    init = rng.uniform(0.5, 1.5, 2 + NUM_PSITES)
    return np.asarray(TIME_POINTS, float), np.asarray(TIME_POINTS_RNA, float), params, init


def _direct_rna(params, init, t0, rna_times):
    from networkmodel.backend import DiffraxSolverConfig, solve_diffrax
    from protwise.models.diffrax_solver import make_local_model_rhs

    grid = np.concatenate([[t0], rna_times]) if rna_times[0] > t0 else rna_times
    sol = np.asarray(solve_diffrax(init, grid, params=params, rhs=make_local_model_rhs(MODEL, NUM_PSITES),
                                   config=DiffraxSolverConfig()))
    return sol[-rna_times.size:, 0]


def test_grids_differ_so_the_test_is_meaningful():
    t, r, _, _ = _setup()
    assert not np.array_equal(t[-r.size:], r)


def test_solve_protwise_ode_reads_rna_at_rna_times():
    from protwise.models.diffrax_solver import solve_protwise_ode

    t, r, params, init = _setup()
    sol_out, flat = solve_protwise_ode(params, init, NUM_PSITES, t, model_name=MODEL)
    assert sol_out.shape == (t.size, 2 + NUM_PSITES)
    expected = np.clip(_direct_rna(params, init, t[0], r), 0, None)
    np.testing.assert_allclose(flat[:r.size], expected, rtol=1e-6, atol=1e-9)


def test_objective_reads_rna_at_rna_times():
    import jax.numpy as jnp

    from protwise.paramest.normest import REGULARIZATION_WEIGHT, protwise_objective, to_opt_space
    from config.constants import USE_REGULARIZATION

    t, r, params, init = _setup()
    target = {"mrna": jnp.asarray(_direct_rna(params, init, t[0], r)),
              "protein": jnp.zeros(t.size), "phospho": jnp.zeros(NUM_PSITES * t.size)}
    mode = {"fit_mrna": True, "fit_protein": False, "fit_phospho": False, "n_rna": r.size,
            "scale_mrna": 1.0, "scale_protein": 1.0, "scale_phospho": 1.0}
    theta = to_opt_space(params, MODEL)
    val = float(protwise_objective(theta, target, init, NUM_PSITES, t, mode, MODEL))
    reg = REGULARIZATION_WEIGHT * float(np.mean(params * params)) if USE_REGULARIZATION else 0.0
    assert abs(val - reg) < 1e-10
