"""Pre-merge item 2: networkmodel.io.load_data's wide-MS branch (x1, x2, ... columns after scaling), where 897615a
added the missing numpy import. process_and_scale_raw_data always returns tidy (time, fc) data, so the real pipeline
does not reach this branch; the test reaches it by replacing the scaler with one that keeps the wide layout."""
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_wide_ms_branch_maps_x_columns_to_the_protein_time_grid(tmp_path, monkeypatch):
    import networkmodel.io as io_mod

    monkeypatch.setattr(io_mod, "logger", logging.getLogger("test_networkmodel_io_wide_ms"))
    kin = tmp_path / "kin.csv"
    pd.DataFrame({"GeneID": ["EGFR"], "Psite": ["Y_1068"], "Kinase": ["{SRC}"]}).to_csv(kin, index=False)
    tf = tmp_path / "tf.csv"
    pd.DataFrame({"Source": ["ELK1"], "Target": ["FOS"]}).to_csv(tf, index=False)
    ms = tmp_path / "ms.csv"
    pd.DataFrame({"GeneID": ["egfr", "EGFR"], "Psite": ["", "Y_1068"],
                  "x1": [1.0, 2.0], "x2": [3.0, 4.0], "x3": [5.0, np.nan]}).to_csv(ms, index=False)

    seen = {}

    def keep_wide(df, time_points, id_cols, scale_method=None, report=None, **kw):
        seen.setdefault("calls", []).append(id_cols)
        return df  # wide layout kept: the x-column branch of load_data runs

    monkeypatch.setattr(io_mod, "process_and_scale_raw_data", keep_wide)
    args = SimpleNamespace(kinase_net=str(kin), tf_net=str(tf), ms=str(ms), rna="", phospho="", kinopt=None,
                           tfopt=None, scaling_method="raw", time_points_protein=np.array([0.0, 2.5, 7.5]))
    _kin, _tf, df_prot, df_pho, df_rna, *_ = io_mod.load_data(args)

    assert seen["calls"] == [["GeneID", "Psite"]]
    assert df_prot.to_dict("list") == {"protein": ["EGFR"] * 3, "time": [0.0, 2.5, 7.5], "fc": [1.0, 3.0, 5.0]}
    assert df_pho.to_dict("list") == {"protein": ["EGFR"] * 2, "psite": ["Y_1068"] * 2, "time": [0.0, 2.5],
                                      "fc": [2.0, 4.0]}  # NaN at x3 dropped
    assert df_rna.empty
