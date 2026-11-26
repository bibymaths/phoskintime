"""N11: the post-fit steady-state log checks each protein's own states over the last time points.
Y from simulate_until_steady has shape (n_times, n_states); the runner indexed Y[i, -window:]."""
from types import SimpleNamespace

import numpy as np
import pytest

N11 = pytest.mark.xfail(reason="N11: steady-state check on the wrong axis", strict=True)


@N11
def test_tail_report_uses_time_axis_and_protein_states(monkeypatch):
    import networkmodel.config as nm_config
    from networkmodel.SteadyStateAnalysis import steady_state_tail_report

    monkeypatch.setattr(nm_config, "MODEL", 0)
    idx = SimpleNamespace(proteins=["A", "B"], offset_y=np.array([0, 3]), n_sites=np.array([1, 2]),
                          n_states=np.array([2, 4]))
    t = np.linspace(0, 1, 50)
    Y = np.ones((t.size, 7))
    Y[:, 3:] = 1.0 + t[:, None]  # protein B (states 3..6) still rising at the end
    Y[:, 1] = 2.0  # protein A total = P0 (2.0) + site (1.0)
    rep = {r["protein"]: r for r in steady_state_tail_report(Y, idx, window=10, tol=1e-4)}
    assert rep["A"]["reached"] and rep["A"]["rel_tail_change"] == 0.0
    assert rep["A"]["final_total_protein"] == pytest.approx(3.0)
    assert not rep["B"]["reached"]
    assert rep["B"]["final_total_protein"] == pytest.approx(3 * 2.0)
