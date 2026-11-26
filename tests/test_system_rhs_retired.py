"""N10: System.rhs is retired; diagnostics use the JAX RHS (jax_rhs_value); numba kernels stay as test oracles."""
import inspect

import pytest


def test_system_rhs_raises():
    from networkmodel.network import System

    with pytest.raises(NotImplementedError, match="N10"):
        System.rhs(object(), 0.0, None)


def test_steady_state_uses_jax_rhs():
    import networkmodel.SteadyStateAnalysis as ss

    src = inspect.getsource(ss)
    assert "jax_rhs_value(sys, t_last, y_last)" in src and "sys.rhs(" not in src


def test_numba_kernels_still_available_as_oracles():
    from networkmodel.models import combinatorial_rhs, distributive_rhs, sequential_rhs  # noqa: F401
