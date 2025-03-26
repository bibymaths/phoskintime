import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def dummy_time_points():
    return np.linspace(0, 10, 5)


@pytest.fixture
def dummy_parameters():
    return {
        "A": 1.0,
        "B": 1.0,
        "C": 1.0,
        "D": 1.0,
        "Ssite": 0.5,
        "Dsite": 0.5
    }


@pytest.fixture
def dummy_bounds():
    return {
        "A": (0, 10),
        "B": (0, 10),
        "C": (0, 10),
        "D": (0, 10),
        "Ssite": (0, 1),
        "Dsite": (0, 1)
    }


@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        "Gene": ["GENE1"],
        "P_1": [0.1],
        "P_2": [0.2],
        "P_3": [0.3],
        "P_4": [0.4],
        "P_5": [0.5],
    })
