"""N12: the PINN/NeuralODE path is quarantined: not imported by the default workflow, refused by the runner."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_workflow_does_not_import_pinn():
    code = ("import sys, networkmodel.runner as r; r._import_runtime_dependencies(); "
            "import networkmodel.simulate, networkmodel.SteadyStateAnalysis, networkmodel.OptimizationProblem; "
            "print(sorted(m for m in sys.modules if 'pinn' in m.split('.')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_runner_refuses_enable_pinn():
    import inspect

    import networkmodel.runner as r

    src = inspect.getsource(r.main)
    i = src.index("if args.enable_pinn:")
    assert "raise ValueError" in src[i:i + 600] and "N12" in src[i:i + 600]
