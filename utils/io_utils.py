import os

import pandas as pd
from constants import OUT_DIR
from core import get_param_names


def ensure_output_directory(directory):
    os.makedirs(directory, exist_ok=True)

def load_data(excel_file, sheet="Estimated Values"):
    return pd.read_excel(excel_file, sheet_name=sheet)

def save_result(results, time_points, excel_filename=os.path.join(OUT_DIR, 'results.xlsx')):
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for res in results:
            gene = res["gene"]
            estimated_params = res["estimated_params"]
            errors = res["errors"]
            param_labels = get_param_names(len(estimated_params[0]))
            rows = []
            for i in range(len(estimated_params)):
                row = {"Gene": gene, "Time": time_points[i]}
                for label, value in zip(param_labels, estimated_params[i]):
                    row[label] = value
                row["Error"] = errors[i] if i < len(errors) else ""
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=gene, index=False)