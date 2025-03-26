
import numpy as np
from plotting.plotting import plot_param_series


def test_plot_param_series(tmp_path):
    params = [np.random.rand(6) for _ in range(10)]
    time = np.linspace(0, 9, 10)
    param_names = ["A", "B", "C", "D", "Ssite1", "Dsite1"]
    gene_name = "TestGene"

    plot_param_series(gene_name, params, param_names, time, tmp_path)

    output_file = tmp_path / f"{gene_name}_params_.png"
    assert output_file.exists()