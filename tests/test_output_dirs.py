"""X11: workflow output (logs, protwise diagrams/sensitivity plots) follows the run's output directory.

No hard-coded shared folder may be created or written when a run redirects its logs and passes
its own output directory.
"""
import logging

import pytest

def test_file_log_directory_is_created_lazily(tmp_path):
    from config.logconf import setup_logger

    log_dir = tmp_path / "default_logs"
    logger = setup_logger(name="x11_lazy_test", log_dir=str(log_dir))
    assert not log_dir.exists(), "setting up a logger must not create its log directory"
    logger.info("first message")
    for h in logger.handlers:
        h.flush()
    assert log_dir.is_dir()
    assert any(log_dir.iterdir())


def test_redirect_file_logs_moves_existing_and_future_loggers(tmp_path):
    from config.logconf import redirect_file_logs, setup_logger

    default_dir = tmp_path / "shared_default"
    run_dir = tmp_path / "run1" / "logs"
    before = setup_logger(name="x11_before_redirect", log_dir=str(default_dir))
    redirect_file_logs(run_dir, [default_dir])
    after = setup_logger(name="x11_after_redirect", log_dir=str(default_dir))
    explicit = setup_logger(name="x11_explicit_dir", log_dir=str(tmp_path / "explicit"))
    for lg in (before, after, explicit):
        lg.info("hello")
        for h in lg.handlers:
            h.flush()

    assert not default_dir.exists()
    names = sorted(p.name for p in run_dir.iterdir())
    assert any(n.startswith("x11_before_redirect_") for n in names)
    assert any(n.startswith("x11_after_redirect_") for n in names)
    assert any((tmp_path / "explicit").iterdir()), "an unrelated log_dir is not redirected"
    file_handlers = [h for h in before.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_illustrate_writes_to_given_out_dir(tmp_path):
    pytest.importorskip("graphviz")
    import inspect

    from protwise.models.diagram import illustrate

    assert "out_dir" in inspect.signature(illustrate).parameters
    try:
        illustrate("GENEX", 2, out_dir=tmp_path)
    except Exception as exc:  # graphviz binary missing
        if "ExecutableNotFound" in type(exc).__name__:
            pytest.skip("graphviz 'dot' executable not available")
        raise
    assert list(tmp_path.glob("GENEX_phospho_diagram*.png"))


def test_sensitivity_analysis_accepts_out_dir():
    import inspect

    from protwise.sensitivity import sensitivity_analysis

    assert "out_dir" in inspect.signature(sensitivity_analysis).parameters
