import os

import pandas as pd
from phoskintime.utils.io_utils import ensure_output_directory, save_result


def test_ensure_output_directory(tmp_path):
    dir_path = tmp_path / "new_dir"
    ensure_output_directory(dir_path)
    assert dir_path.exists()

def test_save_result(tmp_path):
    data = {"Gene": ["Test"], "Score": [1.23]}
    df = pd.DataFrame(data)
    out_dir = tmp_path
    save_result(df, f"{out_dir}/test_result", "test")
    assert os.path.exists(f"{out_dir}/test_result_test.xlsx")
    assert os.path.exists(f"{out_dir}/test_result_test.csv")
