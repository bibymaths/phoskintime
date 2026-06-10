from __future__ import annotations

import json
from dataclasses import dataclass

from dashboard.result_parser import ResultInventory


@dataclass(frozen=True)
class WorkflowDescriptor:
    key: str
    label: str
    description: str


WORKFLOWS: dict[str, WorkflowDescriptor] = {
    "kinopt.local": WorkflowDescriptor("kinopt.local", "KinOpt local", "Local kinase-phosphorylation optimisation results."),
    "tfopt.local": WorkflowDescriptor("tfopt.local", "TFOpt local", "Local transcription-factor optimisation results."),
    "protwise.runner": WorkflowDescriptor("protwise.runner", "ProtWise", "ProtWise ODE modelling results."),
    "networkmodel.runner": WorkflowDescriptor("networkmodel.runner", "Network model", "Integrated global network model results."),
    "unknown": WorkflowDescriptor("unknown", "Unknown workflow", "Result directory without recognised workflow metadata."),
}


def infer_workflow(inventory: ResultInventory) -> WorkflowDescriptor:
    """Infer workflow identity from metadata first, then legacy filenames."""
    if inventory.metadata is not None:
        try:
            metadata = json.loads(inventory.metadata.read_text(encoding="utf-8"))
            workflow = str(metadata.get("workflow", "")).strip()
            if workflow in WORKFLOWS:
                return WORKFLOWS[workflow]
        except (OSError, json.JSONDecodeError):
            pass

    names = {item.name for item in inventory.tables}
    if "kinopt_results.xlsx" in names:
        return WORKFLOWS["kinopt.local"]
    if "tfopt_results.xlsx" in names:
        return WORKFLOWS["tfopt.local"]
    if {"scalar_objective.csv", "pred_prot_picked.csv", "pred_rna_picked.csv", "pred_phospho_picked.csv"} & names:
        return WORKFLOWS["networkmodel.runner"]
    return WORKFLOWS["unknown"]


def registered_workflows() -> list[WorkflowDescriptor]:
    """Return known workflow descriptors for display or tests."""
    return [WORKFLOWS[key] for key in sorted(WORKFLOWS)]
