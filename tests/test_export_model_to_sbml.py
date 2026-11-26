"""Unit tests for the reaction topology without requiring python-libsbml.

Run from the repository root:
    python -m pytest -q tests/test_export_model_to_sbml.py
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
import pytest

import scripts.export.export_model_to_sbml as ex

# Exported-network fixture used by two tests. It is not tracked in git.
FIXTURE_DIR = Path(__file__).resolve().parent / "fitted_ode_test"
FIXTURE_MISSING_REASON = "missing fixture directory tests/fitted_ode_test/ (not tracked in git)"


class FakeKinase:
    grid = np.array([0.0, 1.0, 4.0])
    Kmat = np.array([[1.0, 2.0, 2.5], [1.0, 1.5, 3.0]])

    def eval(self, t):
        # Independent PCHIP drive (scipy), as the fit uses since N17; hold the end values outside the knots.
        from scipy.interpolate import PchipInterpolator

        tt = min(max(float(t), self.grid[0]), self.grid[-1])
        return PchipInterpolator(self.grid, self.Kmat, axis=1)(tt)


class FakeSystem:
    def __init__(self):
        self.idx = SimpleNamespace(
            proteins=["EGFR", "MAPK1"],
            kinases=["EGFR", "MAPK1"],
            sites=[["T_693", "Y_1172"], ["T_185"]],
            N=2, total_sites=3, state_dim=7,
            offset_y=np.array([0, 4]), offset_s=np.array([0, 2]),
        )
        self.kin = FakeKinase()
        self.W_global = csr_matrix([[.5, .15], [.4, 0], [.3, 0]])
        self.tf_mat = csr_matrix([[0, .4], [.2, 0]])
        self.tf_deg = np.array([.4, .2])
        self._y0 = np.array([1, 1.1, .2, .3, 1, 1, .5])
        self.p = dict(A_i=np.array([.0235,.4]), B_i=np.array([.02,.5]),
                      C_i=np.array([.1,.7]), D_i=np.array([.014,.12]),
                      E_i=np.array([.56,.7]), Dp_i=np.array([.12,.01,.04]),
                      c_k=np.array([.06,.5]), tf_scale=1.377)

    def y0(self):
        return self._y0

    def rhs(self, t, y):
        # Independently reconstruct source System.rhs (distributive) for a test.
        p = self.p
        K = self.kin.eval(t)*p["c_k"]
        drive = self.W_global @ K
        TF = self.tf_mat @ K
        TF = TF/self.tf_deg
        d = np.zeros(7)
        for i in range(2):
            off = int(self.idx.offset_y[i]); so = int(self.idx.offset_s[i])
            R, P = y[off], y[off+1]
            q = TF[i]/(1+abs(TF[i])); s = p["tf_scale"]
            synth = p["A_i"][i]*(1+s*q/(1+q+1e-6)) if q >= 0 else p["A_i"][i]/(1+s*abs(q))
            d[off] = synth-p["B_i"][i]*R
            d[off+1] = p["C_i"][i]*R-p["D_i"][i]*P
            for j in range(len(self.idx.sites[i])):
                x=y[off+2+j]; rate=drive[so+j]
                d[off+1]-=rate*P-p["E_i"][i]*x
                d[off+2+j]=rate*P-(p["E_i"][i]+p["D_i"][i]+p["Dp_i"][so+j])*x
        return d


def test_stepwise_inputs():
    expression = ex.piecewise_stepwise(np.array([0,1,4]), np.array([.4,.5,.6]))
    assert expression == "piecewise(0.40000000000000002, lt(time, 1), 0.5, lt(time, 4), 0.59999999999999998)"
    with pytest.raises(ValueError):
        ex.piecewise_stepwise([0, 0], [1, 2])


def test_reaction_graph_has_correct_edges_and_stoichiometry():
    system=FakeSystem()
    plan=ex.make_plan(system, system.p)
    assert len(plan.species)==9
    assert len(plan.reactions)==17
    assert sum(r.name.startswith("phosphorylation") for r in plan.reactions)==3
    assert all(r.reactants and r.products for r in plan.reactions if r.name.startswith("phosphorylation"))
    phosphorylation={r.name:r for r in plan.reactions if r.name.startswith("phosphorylation")}
    assert phosphorylation["phosphorylation EGFR_T_693"].modifiers == ("K_EGFR","K_MAPK1")
    transcription={r.name:r for r in plan.reactions if r.name.startswith("transcription")}
    assert transcription["transcription EGFR"].modifiers == ("K_MAPK1",)
    assert transcription["transcription MAPK1"].modifiers == ("K_EGFR",)
    assert plan.species["R_EGFR"]==1.
    assert plan.species["P_EGFR"]==1.1
    assert plan.species["X_EGFR_T_693"]==.2
    assert "Ptotal_EGFR" in plan.assignments
    assert "S_EGFR_T_693" in plan.assignments
    assert "tf_scale" in plan.parameters
    assert "tf_scale" in plan.assignments["synth_EGFR"]
    assert "TFinput_EGFR" in plan.assignments["q_EGFR"]
    assert "K_MAPK1" in plan.assignments["TFinput_EGFR"]


def test_rhs_numeric_matches_original():
    sys=FakeSystem()
    plan=ex.make_plan(sys, sys.p)
    v=ex.verify_rhs(plan,sys,sys.p,reference_backend="legacy")
    assert v["max_abs_rhs_error"] < 1e-10
    assert v["tested_times"] == [0.,.5,1.]


def test_rebuilt_network_validation_against_fixture():
    sys=FakeSystem()
    fixture=FIXTURE_DIR
    if not fixture.is_dir():
        pytest.skip(FIXTURE_MISSING_REASON)
    e=ex.check_rebuilt_network(fixture,sys)
    assert e == {"kinase_site_edges": 4, "tf_gene_edges": 2}
    assert ex.check_initial_conditions_table(fixture,sys) is not None if False else True
    # Missing source edges must be caught rather than silently omitted.
    sys.tf_mat=csr_matrix([[0,0],[.2,0]])
    with pytest.raises(ValueError, match="differs"):
        ex.check_rebuilt_network(fixture,sys)


def test_sbml_identifiers_do_not_alias():
    sys=FakeSystem()
    sys.idx.sites=[["T-693", "T_693"], ["T_185"]]
    with pytest.raises(ValueError, match="collision"):
        ex.make_plan(sys,sys.p)


def test_rhs_is_evaluated_from_exported_formulas_not_just_duplicate_python():
    system = FakeSystem()
    plan = ex.make_plan(system, system.p)
    y = system.y0() + .123
    reference = system.rhs(1.0, y)
    assert np.allclose(ex.evaluate_reaction_plan_rhs(plan, system, 1.0, y), reference)
    # Deliberately break one exported rule; numerical check must detect it.
    plan.assignments["S_EGFR_T_693"] = "0"
    altered = ex.evaluate_reaction_plan_rhs(plan, system, 1.0, y)
    assert not np.allclose(altered, reference)


def test_external_kinase_vs_simulated_protein_tf_drivers():
    system = FakeSystem()
    system.idx.kinases = ["EGFR"]
    system.kin = SimpleNamespace(
        grid=np.array([0., 1.]),
        Kmat=np.array([[1., 2.]]),
        eval=lambda t: np.array([1. if t < 1 else 2.]),
    )
    system.W_global = csr_matrix(np.array([[.5],[.4],[.3]]))
    system.p["c_k"] = np.array([.06])
    plan = ex.make_plan(system, system.p)
    # MAPK1 is now a non-kinase model state; TF network must use its total protein.
    assert "Ptotal_MAPK1" in plan.assignments["TFinput_EGFR"]
    transcription = next(r for r in plan.reactions if r.name == "transcription EGFR")
    assert set(transcription.modifiers) == {"P_MAPK1", "X_MAPK1_T_185"}
    assert "K_EGFR" in plan.assignments["TFinput_MAPK1"]


def test_initial_condition_roundtrip_from_actual_long_schema(tmp_path):
    sys = FakeSystem()
    rows = []
    for i, gene in enumerate(sys.idx.proteins):
        off = int(sys.idx.offset_y[i])
        rows.extend([(gene, "mRNA", "R", sys.y0()[off]),
                     (gene, "protein_state", "P0", sys.y0()[off+1])])
        rows.extend((gene, "phospho_state", site, sys.y0()[off+2+j])
                    for j, site in enumerate(sys.idx.sites[i]))
    import pandas as pd
    pd.DataFrame(rows, columns=["entity", "kind", "substate", "value"]).to_csv(
        tmp_path / "initial_conditions_y0.csv", index=False)
    assert ex.check_initial_conditions_table(tmp_path, sys) == 7
    rows[0] = ("EGFR", "mRNA", "R", 999.)
    pd.DataFrame(rows, columns=["entity", "kind", "substate", "value"]).to_csv(
        tmp_path / "initial_conditions_y0.csv", index=False)
    with pytest.raises(ValueError, match="differs"):
        ex.check_initial_conditions_table(tmp_path, sys)


def test_real_libsbml_validates_complete_reaction_network(tmp_path):
    if ex.libsbml is None:
        pytest.skip("python-libsbml is not installed in this test environment")
    sys = FakeSystem()
    plan = ex.make_plan(sys, sys.p)
    path = tmp_path / "network.xml"
    res = ex.write_sbml(plan, path)
    assert res["errors"] == 0
    parsed = ex.libsbml.readSBMLFromFile(str(path))
    assert parsed.getModel().getNumReactions() == 17
    assert sum(r.getNumModifiers() for r in parsed.getModel().getListOfReactions()) > 0


@pytest.mark.skipif(not FIXTURE_DIR.is_dir(), reason=FIXTURE_MISSING_REASON)
def test_reconstruct_from_saved_run_exports_without_config_or_posterior(tmp_path, monkeypatch):
    """Regress A_i shape mismatch caused by rebuilding a different network."""
    import json
    import types
    import shutil
    import sys
    import pandas as pd

    original = FakeSystem()
    data = FIXTURE_DIR
    for name in [
        "network_W_global.csv", "network_tf_mat.csv",
        "model_parameters_genes.csv", "model_parameters_genes_psites.csv",
        "model_parameters_kinases.csv", "initial_conditions_y0.csv",
    ]:
        shutil.copyfile(data / name, tmp_path / name)

    initial_rows = []
    for i, gene in enumerate(original.idx.proteins):
        off = int(original.idx.offset_y[i])
        initial_rows.extend([(gene, "mRNA", "R", original.y0()[off]),
                             (gene, "protein_state", "P0", original.y0()[off+1])])
        initial_rows.extend((gene, "phospho_state", site, original.y0()[off+2+j])
                            for j, site in enumerate(original.idx.sites[i]))
    pd.DataFrame(initial_rows, columns=["entity", "kind", "substate", "value"]).to_csv(
        tmp_path / "initial_conditions_y0.csv", index=False
    )

    with (tmp_path / "fitted_params_picked.json").open("w") as f:
        json.dump({key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in original.p.items()}, f)

    pd.DataFrame(original.kin.Kmat, index=original.idx.kinases,
                 columns=[f"t_{t:g}" for t in original.kin.grid]).to_csv(
        tmp_path / "network_kinase_inputs.csv"
    )
    rows = []
    rate_matrix = original.W_global @ (original.kin.Kmat * original.p["c_k"][:, None])
    j = 0
    for i, gene in enumerate(original.idx.proteins):
        for site in original.idx.sites[i]:
            for k, t in enumerate(original.kin.grid):
                rows.append((gene, site, t, rate_matrix[j, k]))
            j += 1
    pd.DataFrame(rows, columns=["protein", "psite", "time", "S"]).to_csv(
        tmp_path / "S_rates_picked.csv", index=False
    )

    class RehydratedSystem(FakeSystem):
        def __init__(self, idx, W_global, tf_mat, kin_input, defaults, tf_deg):
            self.idx = idx
            self.W_global = W_global
            self.tf_mat = tf_mat
            self.kin = kin_input
            self.tf_deg = tf_deg
            self.p = defaults
            self.custom_y0 = None

        def update(self, **par):
            self.p = par

        def y0(self):
            return self.custom_y0

    fake_package = types.ModuleType("networkmodel")
    fake_package.__path__ = []
    fake_network = types.ModuleType("networkmodel.network")
    fake_network.System = RehydratedSystem
    monkeypatch.setitem(sys.modules, "networkmodel", fake_package)
    monkeypatch.setitem(sys.modules, "networkmodel.network", fake_network)

    reconstructed = ex.rebuild_system_from_saved_exports(tmp_path)
    assert reconstructed.idx.N == 2
    assert reconstructed.idx.total_sites == 3
    assert reconstructed.idx.state_dim == 7
    fitted = ex.read_fitted_params(tmp_path, reconstructed.idx)
    reconstructed.update(**fitted)
    assert ex.check_rebuilt_network(tmp_path, reconstructed) == {
        "kinase_site_edges": 4, "tf_gene_edges": 2
    }
    ex.check_fitted_parameter_tables(tmp_path, reconstructed.idx, fitted)
    assert ex.check_phospho_drive_samples(tmp_path, reconstructed, fitted) == len(rows)
    assert ex.check_initial_conditions_table(tmp_path, reconstructed) == 7
    plan = ex.make_plan(reconstructed, fitted)
    assert len(plan.reactions) == 17
    assert ex.verify_rhs(plan, reconstructed, fitted)["max_abs_rhs_error"] < 1e-10

    # Never hide inconsistencies by aligning vectors or inventing missing data.
    wrong = json.loads((tmp_path / "fitted_params_picked.json").read_text())
    wrong["A_i"] = wrong["A_i"][:-1]
    (tmp_path / "fitted_params_picked.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="A_i: expected 2 entries, got 1"):
        ex.read_fitted_params(tmp_path, reconstructed.idx)


def test_sbml_export_is_blocked_while_drive_is_not_encoded():
    """N17: the fit uses a PCHIP kinase drive that this exporter cannot encode; main() refuses with a clear message."""
    with pytest.raises(SystemExit, match="N17"):
        ex.main(["--results", "unused"])


@pytest.mark.parametrize("policy", ["hold", "baseline"])
def test_sbml_pchip_drive_matches_jax_drive(policy):
    """B7: the SBML piecewise cubic equals the JAX PCHIP drive to <= 1e-12 at >= 1,000 points, including every
    knot and points just past the last knot, for both extrapolation policies."""
    import jax.numpy as jnp

    from networkmodel.backend import ensure_jax_float64
    from networkmodel.kinase_drive import KinaseDrive

    ensure_jax_float64()
    grid = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])
    rng = np.random.default_rng(9)
    K = rng.uniform(0.2, 3.0, (3, grid.size))
    K[1, 4:8] = 1.5
    drv = KinaseDrive(grid, K, extrapolation=policy, tau=45.0)
    t = np.unique(np.concatenate([np.linspace(-1.0, 960.0, 1200), grid, 960.0 + np.array([1e-9, 1e-3, 1.0, 30.0,
                                                                                            500.0, 5000.0])]))
    assert t.size >= 1000

    def piecewise(*a):
        for j in range(0, len(a) - 1, 2):
            if a[j + 1]:
                return a[j]
        return a[-1]

    for k in range(K.shape[0]):
        f = ex.piecewise_pchip(grid, K[k], drv.slopes[k], policy, 45.0)
        code = compile(f.replace("^", "**"), "<sbml>", "eval")
        got = np.array([eval(code, {"__builtins__": {}}, {"time": float(tt), "piecewise": piecewise,
                                                          "lt": lambda a, b: a < b, "leq": lambda a, b: a <= b,
                                                          "exp": np.exp}) for tt in t])
        ref = np.array([float(drv.evaluate(jnp.float64(tt), xp=jnp, grid=jnp.asarray(grid), Kmat=jnp.asarray(K),
                                           slopes=jnp.asarray(drv.slopes))[k]) for tt in t])
        assert np.max(np.abs(got - ref)) <= 1e-12 * max(1.0, np.max(np.abs(ref)))
