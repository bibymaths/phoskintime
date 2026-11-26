"""PERF-1: repeated ODE solves reuse compiled code instead of recompiling (issue X13)."""
import numpy as np
import pytest

from protwise.models.diffrax_solver import make_local_model_rhs, solve_protwise_ode


@pytest.mark.parametrize("model,n", [("distmod", 1), ("succmod", 2), ("randmod", 3), ("distributive", 2)])
def test_local_rhs_factory_returns_same_object(model, n):
    assert make_local_model_rhs(model, n) is make_local_model_rhs(model, n)


def test_local_rhs_factory_distinguishes_models_and_sites():
    assert make_local_model_rhs("distmod", 1) is not make_local_model_rhs("distmod", 2)
    assert make_local_model_rhs("distmod", 2) is not make_local_model_rhs("succmod", 2)
    # aliases resolve to the canonical model and share the cached object
    assert make_local_model_rhs("dist", 2) is make_local_model_rhs("distmod", 2)


def test_local_rhs_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported local ODE model"):
        make_local_model_rhs("nope", 1)


def test_repeated_protwise_solve_is_identical():
    t = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0], dtype=float)
    p = np.asarray([0.3, 0.2, 0.35, 0.1, 0.18, 0.12], dtype=float)
    y0 = np.asarray([1.0, 1.0, 1.0], dtype=float)
    a, fa = solve_protwise_ode(p, y0, 1, t, model_name="distmod")
    b, fb = solve_protwise_ode(p, y0, 1, t, model_name="distmod")
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(fb))


# --- PERF-1: persistent compilation cache configuration -------------------------------------

def test_cache_dir_resolution(monkeypatch, tmp_path):
    from common import jax_cache

    monkeypatch.delenv(jax_cache.JAX_CACHE_ENV, raising=False)
    assert jax_cache.resolve_cache_dir() == jax_cache.DEFAULT_JAX_CACHE_DIR
    for off in ("off", "0", "false", "none", ""):
        monkeypatch.setenv(jax_cache.JAX_CACHE_ENV, off)
        assert jax_cache.resolve_cache_dir() is None
    monkeypatch.setenv(jax_cache.JAX_CACHE_ENV, str(tmp_path / "c"))
    assert jax_cache.resolve_cache_dir() == tmp_path / "c"


def test_configure_cache_sets_jax_config(tmp_path):
    import jax
    from common import jax_cache

    old_dir = jax.config.jax_compilation_cache_dir
    old_min = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        assert jax_cache.configure_persistent_compilation_cache("off") is None
        used = jax_cache.configure_persistent_compilation_cache(tmp_path / "jc")
        assert used == tmp_path / "jc" and used.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "jc")
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    finally:
        jax.config.update("jax_compilation_cache_dir", old_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", old_min)
