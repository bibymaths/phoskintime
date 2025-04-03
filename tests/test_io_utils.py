import os
import pytest
import pandas as pd
from utils.display import ensure_output_directory, save_result

def ensure_output_directory_creates_nested_directories(tmp_path):
    nested_dir = tmp_path / "nested" / "subdir" / "results"
    ensure_output_directory(nested_dir)
    assert nested_dir.exists()

def save_result_creates_file_with_multiple_gene_sheets(tmp_path):
    gene_a = "GeneA"
    gene_b = "GeneB"
    param_df_a = pd.DataFrame({"param_1": [1, 2], "param_2": [3, 4]})
    param_df_b = pd.DataFrame({"param_1": [5, 6], "param_2": [7, 8]})
    results = [
        {"gene": gene_a, "param_df": param_df_a, "errors": [0.1, 0.2]},
        {"gene": gene_b, "param_df": param_df_b, "errors": [0.3, 0.4]}
    ]
    excel_filename = tmp_path / "multiple_genes.xlsx"
    save_result(results, str(excel_filename))
    assert os.path.exists(excel_filename)
    df_a = pd.read_excel(excel_filename, sheet_name=f"{gene_a}_params")
    df_b = pd.read_excel(excel_filename, sheet_name=f"{gene_b}_params")
    assert not df_a.empty
    assert not df_b.empty

def save_result_handles_empty_results_list(tmp_path):
    excel_filename = tmp_path / "empty_results.xlsx"
    save_result([], str(excel_filename))
    assert os.path.exists(excel_filename)
    with pytest.raises(ValueError):
        pd.read_excel(str(excel_filename))

def ensure_output_directory_passes_existing_directory(tmp_path):
    existing_dir = tmp_path / "existing_directory"
    existing_dir.mkdir()
    ensure_output_directory(existing_dir)
    assert existing_dir.exists()