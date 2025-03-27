import os, re, shutil
import pandas as pd

def ensure_output_directory(directory):
    os.makedirs(directory, exist_ok=True)

def load_data(excel_file, sheet="Estimated Values"):
    return pd.read_excel(excel_file, sheet_name=sheet)

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"


def organize_output_files(*directories):
    protein_regex = re.compile(r'([A-Za-z0-9]+)_.*\.(json|svg|png|html|csv)$')

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: '{directory}' is not a valid directory. Skipping.")
            continue

        # Move files matching the protein pattern.
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                match = protein_regex.search(filename)
                if match:
                    protein = match.group(1)
                    protein_folder = os.path.join(directory, protein)
                    os.makedirs(protein_folder, exist_ok=True)
                    destination_path = os.path.join(protein_folder, filename)
                    shutil.move(file_path, destination_path)

        # After protein files have been moved, move remaining files to a "General" folder.
        general_folder = os.path.join(directory, "General")
        os.makedirs(general_folder, exist_ok=True)
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                destination_path = os.path.join(general_folder, filename)
                shutil.move(file_path, destination_path)


def save_result(results, excel_filename):
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for res in results:
            gene = res["gene"]
            sheet_prefix = gene[:25]  # Excel sheet names must be ≤31 chars

            # 1. Save Sequential Parameter Estimates
            param_df = res["param_df"].copy()
            if "errors" in res:
                param_df["SSE"] = pd.Series(res["errors"][:len(param_df)])
            param_df.insert(0, "Gene", gene)
            param_df.to_excel(writer, sheet_name=f"{sheet_prefix}_params", index=False)

            # 2. Save Profiled Estimates if available
            if "profiles_df" in res and res["profiles_df"] is not None:
                prof_df = res["profiles_df"].copy()
                prof_df.insert(0, "Gene", gene)
                prof_df.to_excel(writer, sheet_name=f"{sheet_prefix}_profiles", index=False)

            # 4. Save errors summary
            error_summary = {
                "Gene": gene,
                "MSE": res.get("mse", ""),
                "MAE": res.get("mae", "")
            }
            err_df = pd.DataFrame([error_summary])
            err_df.to_excel(writer, sheet_name=f"{sheet_prefix}_errors", index=False)
