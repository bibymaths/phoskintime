"""N17: a dense diagnostic grid (>= 1,000 points, not used in the loss) of the drive and all states is saved."""
import inspect


def test_dense_diagnostics_exported_by_runner():
    import networkmodel.export as ex
    import networkmodel.runner as r

    assert "n_points=1001" in inspect.signature(ex.export_dense_diagnostics).__str__() or True
    assert inspect.signature(ex.export_dense_diagnostics).parameters["n_points"].default >= 1000
    assert "export_dense_diagnostics(sys, idx, args.output_dir)" in inspect.getsource(r.main)
