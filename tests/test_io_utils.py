import os
import pandas as pd
import numpy as np
from utils.utils import ensure_output_directory, save_result


def test_ensure_output_directory(tmp_path):
    dir_path = tmp_path / "new_dir"
    ensure_output_directory(dir_path)
    assert dir_path.exists()


def test_save_result(tmp_path):
    # Create dummy result data
    gene = "TestGene"
    estimated_params = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])] * 3
    errors = [0.01, 0.02, 0.03]

    results = [{
        "gene": gene,
        "estimated_params": estimated_params,
        "errors": errors
    }]

    excel_filename = tmp_path / "test_results.xlsx"
    save_result(results, str(excel_filename))

    assert os.path.exists(excel_filename)

    # Validate sheet exists
    df = pd.read_excel(excel_filename, sheet_name=gene)
    assert not df.empty
    assert "Gene" in df.columns
    assert "Time" in df.columns
