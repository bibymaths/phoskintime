from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.results import ensure_result_dir


@dataclass(frozen=True)
class AnalysisTask:
    key: str
    label: str
    description: str
    script: str
    arguments: dict[str, str] = field(default_factory=dict)


ANALYSIS_TASKS: dict[str, AnalysisTask] = {
    "tf-kin-counts": AnalysisTask(
        "tf-kin-counts",
        "TF/Kin counts",
        "Summarise TF and kinase psite/target counts from TFOpt and KinOpt workbooks.",
        "scripts/analyze_tf_kin_counts.py",
        {"tfopt_xlsx": "--tfopt-xlsx", "kinopt_xlsx": "--kinopt-xlsx", "out_dir": "--out-dir"},
    ),
    "curve-similarity": AnalysisTask(
        "curve-similarity",
        "Curve similarity / Fréchet distance",
        "Run the existing curve similarity backend for TFOpt and KinOpt workbooks.",
        "scripts/curve_similarity.py",
        {"tfopt_xlsx": "--tfopt-xlsx", "kinopt_xlsx": "--kinopt-xlsx", "out_dir": "--out-dir"},
    ),
    "export-subnetworks": AnalysisTask(
        "export-subnetworks",
        "Export subnetworks",
        "Run the existing subnetworks export script on kinase and TF network input files.",
        "scripts/export_subnetworks.py",
        {"input2": "--input2", "input4": "--input4", "out_dir": "--outdir", "hops": "--hops"},
    ),
    "protein-accumulators": AnalysisTask(
        "protein-accumulators",
        "Protein accumulators",
        "Detect protein-vs-RNA accumulator patterns from networkmodel prediction CSVs.",
        "scripts/find_protein_accumulators.py",
        {"prot": "--prot", "rna": "--rna", "threshold": "--threshold"},
    ),
    "mechanistic-insights": AnalysisTask(
        "mechanistic-insights",
        "Mechanistic insights",
        "Run the existing mechanistic insights script; inputs are passed through its CLI.",
        "scripts/mechanistic_insights.py",
        {"results_dir": "--results-dir", "out_dir": "--out-dir"},
    ),
    "temporal-sensitivity": AnalysisTask(
        "temporal-sensitivity",
        "Temporal sensitivity",
        "Run global temporal sensitivity analysis for an existing networkmodel results directory.",
        "scripts/temporal_sensitivity.py",
        {"results_dir": "--results-dir", "samples": "--samples"},
    ),
}


def analysis_output_dir(result_dir: str | Path, task_key: str) -> Path:
    """Create a dashboard-standard artifacts/reports area for analysis outputs."""
    paths = ensure_result_dir(result_dir)
    out_dir = paths["artifacts"] / "analysis" / task_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_analysis_command(task_key: str, values: dict[str, Any], result_dir: str | Path) -> list[str]:
    """Build a safe argv-list command for an existing analysis script."""
    if task_key not in ANALYSIS_TASKS:
        raise KeyError(f"Unknown analysis task: {task_key}")
    task = ANALYSIS_TASKS[task_key]
    command = ["python", task.script]
    merged = dict(values)
    if "out_dir" in task.arguments and not merged.get("out_dir"):
        merged["out_dir"] = analysis_output_dir(result_dir, task_key)
    elif task_key in {"protein-accumulators", "temporal-sensitivity"}:
        analysis_output_dir(result_dir, task_key)
    for name, flag in task.arguments.items():
        value = merged.get(name)
        if value is None or value == "":
            continue
        command.extend([flag, str(value)])
    return command


def discover_analysis_outputs(result_dir: str | Path) -> dict[str, list[Path]]:
    """Discover outputs previously written by dashboard-triggered analyses."""
    root = Path(result_dir)
    base = root / "artifacts" / "analysis"
    if not base.is_dir():
        return {}
    return {
        task_dir.name: sorted(path for path in task_dir.rglob("*") if path.is_file())
        for task_dir in sorted(base.iterdir())
        if task_dir.is_dir()
    }


def render(root: str | Path) -> None:
    import streamlit as st

    st.subheader("Advanced analyses")
    st.caption("Analyses are not run automatically. Build a command, review it, and run via the launcher or terminal when appropriate.")
    task = st.selectbox("Analysis", list(ANALYSIS_TASKS.values()), format_func=lambda item: item.label)
    st.write(task.description)
    values: dict[str, Any] = {}
    for name in task.arguments:
        if name == "out_dir":
            continue
        values[name] = st.text_input(name.replace("_", " "), value="")
    try:
        command = build_analysis_command(task.key, values, root)
        st.code(" ".join(command), language="bash")
    except Exception as exc:
        st.error(str(exc))
    outputs = discover_analysis_outputs(root)
    if outputs:
        st.write("Existing analysis outputs")
        st.json({key: [str(path) for path in paths] for key, paths in outputs.items()})
