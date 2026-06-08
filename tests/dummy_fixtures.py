"""Reusable deterministic dummy data for notebook-readiness tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


KINOPT_TIME_COLS = [f"x{i}" for i in range(1, 15)]
TFOPT_TIME_POINTS = np.asarray([4, 8, 15, 30, 60, 120, 240, 480, 960], dtype=float)
PROTWISE_TIME_POINTS = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
NETWORK_TIME_POINTS = np.asarray([0.0, 1.0, 2.0], dtype=float)


def kinopt_dummy_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = np.linspace(1.0, 1.3, len(KINOPT_TIME_COLS))
    rows = [
        {"GeneID": "P1", "Psite": "S1", **dict(zip(KINOPT_TIME_COLS, base))},
        {"GeneID": "P2", "Psite": "S1", **dict(zip(KINOPT_TIME_COLS, base * 0.9 + 0.1))},
        {"GeneID": "K1", "Psite": "T100", **dict(zip(KINOPT_TIME_COLS, base * 1.1))},
        {"GeneID": "K2", "Psite": "T200", **dict(zip(KINOPT_TIME_COLS, base * 0.8 + 0.2))},
    ]
    interactions = pd.DataFrame(
        {
            "GeneID": ["P1", "P2"],
            "Psite": ["S1", "S1"],
            "Kinase": [["K1", "K2"], ["K2"]],
        }
    )
    return pd.DataFrame(rows), interactions


def tfopt_dummy_data():
    gene_ids = ["G1", "G2"]
    tf_ids = ["TF1", "TF2"]
    expression = np.vstack([
        np.linspace(1.0, 1.4, len(TFOPT_TIME_POINTS)),
        np.linspace(0.9, 1.2, len(TFOPT_TIME_POINTS)),
    ])
    tf_protein = {
        "TF1": np.linspace(1.0, 1.1, len(TFOPT_TIME_POINTS)),
        "TF2": np.linspace(0.8, 1.0, len(TFOPT_TIME_POINTS)),
    }
    tf_psite_data = {
        "TF1": [np.linspace(1.0, 1.25, len(TFOPT_TIME_POINTS))],
        "TF2": [np.linspace(0.95, 1.05, len(TFOPT_TIME_POINTS))],
    }
    tf_psite_labels = {"TF1": ["S10"], "TF2": ["T20"]}
    reg_map = {"G1": ["TF1", "TF2"], "G2": ["TF2"]}
    return gene_ids, expression, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, reg_map


def protwise_dummy_series(num_psites: int = 1):
    t = PROTWISE_TIME_POINTS
    mrna = 1.0 + 0.05 * t
    protein = 1.0 + 0.04 * t
    phospho = np.vstack([1.0 + (0.03 + 0.01 * i) * t for i in range(num_psites)]).reshape(-1)
    init = np.r_[mrna[0], protein[0], np.ones(num_psites)]
    return t, mrna, protein, phospho, init


@dataclass
class DummyNetworkIndex:
    N: int = 2

    def __post_init__(self):
        self.p2i = {"P1": 0, "P2": 1}
        self.sites = [["S1"], ["S1"]]
        self.n_sites = np.asarray([1, 1], dtype=np.int32)
        self.n_states = np.asarray([2, 2], dtype=np.int32)

    def block(self, i: int) -> slice:
        start = i * 3
        return slice(start, start + 3)


def networkmodel_dummy_frames(include_rna: bool = True, include_phospho: bool = True):
    prot = pd.DataFrame(
        {
            "protein": ["P1", "P1", "P2", "P2"],
            "time": [0.0, 1.0, 0.0, 1.0],
            "fc": [1.0, 1.1, 1.0, 0.95],
            "w": [1.0, 1.0, 1.0, 1.0],
        }
    )
    rna = pd.DataFrame(
        {
            "protein": ["P1", "P2"],
            "time": [1.0, 1.0],
            "fc": [1.05, 0.98],
        }
    ) if include_rna else pd.DataFrame(columns=["protein", "time", "fc"])
    phospho = pd.DataFrame(
        {
            "protein": ["P1", "P2"],
            "psite": ["S1", "S1"],
            "time": [1.0, 1.0],
            "fc": [1.2, 0.9],
        }
    ) if include_phospho else pd.DataFrame(columns=["protein", "psite", "time", "fc"])
    return DummyNetworkIndex(), prot, rna, phospho, NETWORK_TIME_POINTS
