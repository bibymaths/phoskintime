import os
import pandas as pd
import numpy as np
from utils.display import ensure_output_directory, save_result

def test_ensure_output_directory(tmp_path):
    dir_path = tmp_path / "new_dir"
    ensure_output_directory(dir_path)
    assert dir_path.exists()

def test_save_result(tmp_path):
    # Create dummy result data
    gene = "TestGene"
    # Create a dummy DataFrame for parameters
    param_df = pd.DataFrame({
        "param_1": [0.1, 0.1, 0.1],
        "param_2": [0.2, 0.2, 0.2],
        "param_3": [0.3, 0.3, 0.3],
        "param_4": [0.4, 0.4, 0.4],
        "param_5": [0.5, 0.5, 0.5],
        "param_6": [0.6, 0.6, 0.6],
    })
    errors = [0.01, 0.02, 0.03]

    results = [{
        "gene": gene,
        "param_df": param_df,
        "errors": errors
    }]

    excel_filename = tmp_path / "test_results.xlsx"
    save_result(results, str(excel_filename))

    assert os.path.exists(excel_filename)

    # The module writes the parameter estimates to a sheet named "<gene>_params"
    sheet_name = f"{gene}_params"
    df = pd.read_excel(excel_filename, sheet_name=sheet_name)
    assert not df.empty
    # Check that required columns are present
    assert "Gene" in df.columns
    assert "SSE" in df.columns
