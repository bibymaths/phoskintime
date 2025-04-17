import os
import re
import pandas as pd
import pytest
from utils.display import (
    ensure_output_directory,
    load_data,
    format_duration,
    save_result,
    create_report,
    organize_output_files
)


def test_ensure_output_directory_creates_nested_directories(tmp_path):
    nested_dir = tmp_path / "nested" / "subdir" / "results"
    ensure_output_directory(str(nested_dir))
    assert nested_dir.exists()


def test_format_duration_seconds():
    # Test a duration less than 60 seconds.
    result = format_duration(45.678)
    assert "sec" in result


def test_format_duration_minutes():
    # Test a duration between 60 and 3600 seconds.
    result = format_duration(120)
    assert "min" in result


def test_format_duration_hours():
    # Test a duration greater than or equal to 3600 seconds.
    result = format_duration(7200)
    assert "hr" in result


def test_load_data(tmp_path):
    # Create a dummy Excel file with a sheet named "Estimated Values".
    file_path = tmp_path / "dummy.xlsx"
    df_original = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        df_original.to_excel(writer, sheet_name="Estimated Values", index=False)
    df_loaded = load_data(str(file_path))
    pd.testing.assert_frame_equal(df_original, df_loaded)


def test_save_result_creates_file_with_multiple_gene_sheets(tmp_path):
    gene_a = "GeneA"
    gene_b = "GeneB"
    param_df_a = pd.DataFrame({"param_1": [1, 2], "param_2": [3, 4]})
    param_df_b = pd.DataFrame({"param_1": [5, 6], "param_2": [7, 8]})
    results = [
        {"gene": gene_a, "param_df": param_df_a, "errors": [0.1, 0.2]},
        {"gene": gene_b, "param_df": param_df_b, "errors": [0.3, 0.4]}
    ]
    excel_filename = tmp_path / "results.xlsx"
    save_result(results, str(excel_filename))
    assert os.path.exists(excel_filename)
    df_a = pd.read_excel(excel_filename, sheet_name=f"{gene_a[:25]}_params")
    df_b = pd.read_excel(excel_filename, sheet_name=f"{gene_b[:25]}_params")
    assert not df_a.empty
    assert not df_b.empty


def test_save_result_handles_empty_results_list(tmp_path):
    # When passing an empty results list, an Excel file is still created but reading any sheet should raise an error.
    excel_filename = tmp_path / "empty.xlsx"
    save_result([], str(excel_filename))
    assert os.path.exists(excel_filename)
    with pytest.raises(ValueError):
        pd.read_excel(str(excel_filename))


def test_create_report(tmp_path):
    # Create dummy gene folders with a PNG and XLSX file each.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for gene in ["GENE1", "GENE2"]:
        gene_dir = results_dir / gene
        gene_dir.mkdir()
        # Create a dummy PNG file.
        png_path = gene_dir / f"{gene}_plot.png"
        with open(png_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        # Create a dummy XLSX file.
        df = pd.DataFrame({"C": [9, 8], "D": [7, 6]})
        xlsx_path = gene_dir / f"{gene}_data.xlsx"
        df.to_excel(xlsx_path, index=False)
    create_report(str(results_dir), output_file="test_report.html")
    report_path = results_dir / "test_report.html"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Ensure that the report contains gene section headers.
        assert "Protein Group: GENE1" in content
        assert "Protein Group: GENE2" in content


def test_organize_output_files(tmp_path):
    # Create a temporary directory with files that match the protein pattern and some generic files.
    test_dir = tmp_path / "files"
    test_dir.mkdir()
    # Files matching protein pattern.
    filenames = ["ABC_info.png", "XYZ_plot.svg", "123_data.csv"]
    for fname in filenames:
        with open(test_dir / fname, "w") as f:
            f.write("dummy")
    # Generic file.
    generic_file = "misc.txt"
    with open(test_dir / generic_file, "w") as f:
        f.write("miscellaneous")

    # Call organize_output_files on the test_dir.
    organize_output_files(str(test_dir))

    # Check that files matching protein pattern are moved into a folder named after the protein.
    for fname in filenames:
        match = re.search(r'([A-Za-z0-9]+)_.*\.(json|svg|png|html|csv|xlsx)$', fname)
        if match:
            protein = match.group(1)
            protein_folder = test_dir / protein
            assert protein_folder.exists()
            assert (protein_folder / fname).exists()

    # Check that the generic file is moved to the "General" folder.
    general_folder = test_dir / "General"
    assert general_folder.exists()
    assert (general_folder / generic_file).exists()