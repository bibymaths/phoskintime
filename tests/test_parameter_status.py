"""N22 (P3 decision 5): parameters within 1 % of a design bound are labelled, never presented as point estimates."""
import numpy as np
import pandas as pd

from tests.test_design_bounds import GRIDS



def _inv_sp(x):
    return np.log(np.expm1(x))


def _case():
    from networkmodel.utils import design_rate_bounds

    d = design_rate_bounds(GRIDS)
    g = d["groups"]
    names = ["A_i[P]", "B_i[P]", "D_i[P]", "c_k[K1]", "c_k[K2]", "c_k[K3]", "Dp_i[P_s1]", "tf_scale"]
    slices = {"A_i": slice(0, 1), "B_i": slice(1, 2), "D_i": slice(2, 3), "c_k": slice(3, 6), "Dp_i": slice(6, 7),
              "tf_scale": slice(7, 8)}
    lo_ph = [0.5, g["B_i"]["k_min"], g["D_i"]["k_min"], g["c_k"]["k_min"], g["c_k"]["k_min"], 0.016,
             g["Dp_i"]["k_min"], 0.5]
    hi_ph = [2.0, g["B_i"]["k_max"], g["D_i"]["k_max"], 11.0, 11.0, 0.016, g["Dp_i"]["k_max"], 5.0]
    val = [2.0 * 0.995,                       # A_i: within 1 % of its (non-design) upper bound
           g["B_i"]["k_max"] * 0.995,         # B_i: within 1 % of k_max -> above resolvable rate
           g["D_i"]["k_min"] * 1.005,         # D_i: within 1 % of k_min -> below resolvable rate
           0.3,                               # c_k[K1]: interior -> estimate
           g["c_k"]["k_min"] * 1.001,         # c_k[K2]: at design lower
           0.016,                             # c_k[K3]: fixed (zero influence)
           0.2,                               # Dp_i: interior
           1.0]                               # tf_scale: interior
    return d, names, slices, _inv_sp(np.array(val)), _inv_sp(np.array(lo_ph)), _inv_sp(np.array(hi_ph))


def test_status_and_labels():
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"]).set_index("parameter")
    assert st.loc["A_i[P]", "status"] == "at_bound" and not st.loc["A_i[P]", "is_estimate"]
    assert st.loc["B_i[P]", "status"] == "above_resolvable_rate"
    assert st.loc["B_i[P]", "label"] == "above resolvable rate (half-life < 72.2 s)"
    assert st.loc["D_i[P]", "status"] == "below_resolvable_rate"
    assert st.loc["D_i[P]", "label"] == "below resolvable rate (half-life > 105.3 h)"
    assert st.loc["c_k[K2]", "status"] == "below_resolvable_rate"
    assert st.loc["c_k[K3]", "status"] == "fixed_no_data_influence"
    assert st.loc["c_k[K3]", "label"] == "not identifiable: no data influence"
    for n in ("c_k[K1]", "Dp_i[P_s1]", "tf_scale"):  # pre-merge 5: interior, not assessed without profiles
        assert st.loc[n, "status"] == "interior" and st.loc[n, "label"] == "not assessed"
        assert not st.loc[n, "is_estimate"]
    np.testing.assert_allclose(st.loc["c_k[K1]", "value"], 0.3, rtol=1e-12)


def test_exported_tables_carry_the_labels(tmp_path):
    from networkmodel.export import annotate_parameter_tables
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"])
    pd.DataFrame({"Protein_Gene": ["P"], "Synthesis_A": [1.99], "mRNA_Degradation_B": [0.57],
                  "Protein_Degradation_D": [1e-4]}).to_csv(tmp_path / "model_parameters_genes.csv", index=False)
    pd.DataFrame({"Protein_Gene": ["P"], "psite": ["s1"], "Phospho_Degradation_Dp": [0.2]}).to_csv(
        tmp_path / "model_parameters_genes_psites.csv", index=False)
    pd.DataFrame({"Kinase": ["K1", "K2", "K3"], "Activity_Scale_ck": [0.3, 1e-4, 0.016]}).to_csv(
        tmp_path / "model_parameters_kinases.csv", index=False)
    annotate_parameter_tables(str(tmp_path), st)
    g = pd.read_csv(tmp_path / "model_parameters_genes.csv")
    assert "below resolvable rate (half-life > 105.3 h)" in g.loc[0, "Not_point_estimates"]
    assert "mRNA_Degradation_B: above resolvable rate" in g.loc[0, "Not_point_estimates"]
    k = pd.read_csv(tmp_path / "model_parameters_kinases.csv").set_index("Kinase")
    assert k.loc["K1", "Status"] == "not assessed"
    assert k.loc["K2", "Status"].startswith("below resolvable rate")
    assert k.loc["K3", "Status"] == "not identifiable: no data influence"
    ps = pd.read_csv(tmp_path / "model_parameters_genes_psites.csv")
    assert ps.loc[0, "Dp_status"] == "not assessed"
    assert (tmp_path / "parameter_status.csv").exists()


def test_ck_label_is_a_time_constant_and_prefix_collision():
    """Review D-4 (c_k wording) and D-5 (protein X vs X_2 in the Dp flags)."""
    from networkmodel.export import annotate_parameter_tables
    from networkmodel.utils import parameter_status

    d, names, slices, th, xl, xu = _case()
    st = parameter_status(names, th, xl, xu, slices, d, fixed_names=["c_k[K3]"]).set_index("parameter")
    assert st.loc["c_k[K2]", "label"].startswith("below resolvable rate (time constant 1/c_k > ")
    assert "half-life" not in st.loc["c_k[K2]", "label"]
    st = pd.DataFrame([{"parameter": "Dp_i[X_s1]", "label": "estimate (two-sided)", "is_estimate": True},
                       {"parameter": "Dp_i[X_2_s1]", "label": "below resolvable rate (half-life > 1 h)",
                        "is_estimate": False}])
    import tempfile, os
    with tempfile.TemporaryDirectory() as t:
        pd.DataFrame({"Protein_Gene": ["X", "X_2"], "Protein_Degradation_D": [0.1, 0.1]}).to_csv(
            os.path.join(t, "model_parameters_genes.csv"), index=False)
        annotate_parameter_tables(t, st, {"X": ["s1"], "X_2": ["s1"]})
        g = pd.read_csv(os.path.join(t, "model_parameters_genes.csv")).fillna("")
        assert g.loc[0, "Not_point_estimates"] == ""
        assert "Dp_i[X_2_s1]" in g.loc[1, "Not_point_estimates"]
