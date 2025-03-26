import os

import numpy as np
from plotting.plotting import plot_param_series


def test_plot_param_series(tmp_path):
    params = [np.random.rand(6) for _ in range(10)]
    time = np.linspace(0, 9, 10)
    param_names = ["A", "B", "C", "D", "Ssite1", "Dsite1"]
    plot_param_series("TestGene", params, param_names, time, tmp_path)
    assert os.path.exists(tmp_path / "param_series_TestGene.png")