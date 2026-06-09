"""Reusable deterministic dummy data for notebook-readiness tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PROTWISE_TIME_POINTS = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
NETWORK_TIME_POINTS = np.asarray([0.0, 1.0, 2.0], dtype=float)


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
