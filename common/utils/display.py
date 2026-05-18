import os, re, shutil
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from config.constants import TIME_POINTS, model_type, ODE_MODEL


def ensure_output_directory(directory):
    """
    Ensure the output directory exists. If it doesn't, create it.

    Args:
        directory (str): Path to the output directory.
    """
    os.makedirs(directory, exist_ok=True)


def load_data(excel_file, sheet="Estimated Values"):
    """
    Load data from an Excel file. The default sheet is "Estimated Values".

    Args:
        excel_file (str): Path to the Excel file.
        sheet (str): Name of the sheet to load. Default is "Estimated Values".

    Returns:
        pd.DataFrame: DataFrame containing the data from the specified sheet.
    """
    return pd.read_excel(excel_file, sheet_name=sheet)


def format_duration(seconds):
    """
    Format a duration in seconds into a human-readable string.

    Args:
        seconds (float): Duration in seconds.
    Returns:
        str: Formatted duration string.
    """
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"


def merge_obs_est(filename):
    """
    Function to merge observed and estimated data from an Excel file.

    Args:
        filename (str): Path to the Excel file containing observed and estimated data.

    Returns:
        pd.DataFrame: Merged DataFrame containing observed and estimated values for each gene and Psite.
    """
    xls = pd.ExcelFile(filename)
    all_data = []

    obs_sheets = [s for s in xls.sheet_names if s.endswith("_site_observed")]

    for obs_sheet in obs_sheets:
        gene = obs_sheet.replace("_site_observed", "")
        est_sheet = f"{gene}_site_estimates"

        if est_sheet not in xls.sheet_names:
            continue

        # Load both sheets with Psite as index
        obs_df = pd.read_excel(xls, sheet_name=obs_sheet, index_col=0)
        est_df = pd.read_excel(xls, sheet_name=est_sheet, index_col=0)

        for psite in obs_df.index:
            obs_vals = obs_df.loc[psite].values
            est_vals = est_df.loc[psite].values

            row = {
                "Gene": gene,
                "Psite": psite
            }
            row.update({f"x{i + 1}_obs": val for i, val in enumerate(obs_vals)})
            row.update({f"x{i + 1}_est": val for i, val in enumerate(est_vals)})

            all_data.append(row)

    return pd.DataFrame(all_data)


def save_result(results, excel_filename):
    """
    Function to save results to an Excel file.

    Args:
        results (list): List of dictionaries containing results for each gene.
        excel_filename (str): Path to the output Excel file.

    """
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for res in results:
            gene = res["gene"]
            sheet_prefix = gene[:25]  # Excel sheet names must be ≤31 chars

            # 1. Parameter Estimates
            param_df = res["param_df"].copy()
            param_df.rename(columns={"Time": "Time(min)"}, inplace=True)
            if "errors" in res:
                param_df["MSE"] = pd.Series(res["errors"][:len(param_df)])
            param_df.insert(0, "Gene", gene)
            param_df['Regularization'] = res["regularization"]
            param_df.to_excel(writer, sheet_name=f"{sheet_prefix}_params", index=False)

            # 4. Save errors summary
            error_summary = {
                "Gene": gene,
                "MSE": res.get("mse", ""),
                "MAE": res.get("mae", "")
            }
            err_df = pd.DataFrame([error_summary])
            err_df.to_excel(writer, sheet_name=f"{sheet_prefix}_errors", index=False)

            # 5. Save model fits for the system
            fits_arr = np.array(res["model_fits"])
            state_labels = res.get("labels")
            fits_df = pd.DataFrame(fits_arr, columns=state_labels, index=TIME_POINTS)
            fits_df = fits_df.T
            fits_df.index.name = "States/Time(min)"
            fits_df.to_excel(writer, sheet_name=f"{sheet_prefix}_solution")

            # 6. Save model fits for each site
            seq_arr = np.array(res["seq_model_fit"])
            seq_df = pd.DataFrame(seq_arr, index=res["psite_labels"], columns=TIME_POINTS)
            seq_df.index.name = "Site/Time(min)"
            seq_df.to_excel(writer, sheet_name=f"{sheet_prefix}_site_estimates")

            # 7. Save observed data for each site
            obs_arr = np.array(res["observed_data"])
            obs_df = pd.DataFrame(obs_arr, index=res["psite_labels"], columns=TIME_POINTS)
            obs_df.index.name = "Site/Time(min)"
            obs_df.to_excel(writer, sheet_name=f"{sheet_prefix}_site_observed")

            # 8. Save PCA results
            pca_arr = np.array(res["pca_result"])
            ev = np.array(res["ev"])
            pca_df = pd.DataFrame(pca_arr, columns=[f"PC{i + 1}" for i in range(pca_arr.shape[1])])
            pca_df.insert(0, "Time(min)", TIME_POINTS)
            ev_row = pd.DataFrame([["Explained Var"] + list(ev)], columns=pca_df.columns)
            pca_df = pd.concat([pca_df, ev_row], ignore_index=True)
            pca_df.to_excel(writer, sheet_name=f"{sheet_prefix}_pca", index=False)

            # 9. Save t-SNE results
            tsne_arr = np.array(res["tsne_result"])
            tsne_df = pd.DataFrame(tsne_arr, columns=[f"t-SNE{i + 1}" for i in range(tsne_arr.shape[1])])
            tsne_df.insert(0, "Time(min)", TIME_POINTS)
            tsne_df.to_excel(writer, sheet_name=f"{sheet_prefix}_tsne", index=False)

            # 10. Save knockout results
            system_fits = []
            for knockout_name, ko_data in res["knockout_results"].items():
                sol_ko = np.array(ko_data["sol_ko"])
                state_labels = [f"{ps}" for ps in res["labels"]]
                sys_df = pd.DataFrame(sol_ko, columns=[f"{knockout_name} | {label}" for label in state_labels],
                                      index=TIME_POINTS)
                system_fits.append(sys_df)
            system_concat = pd.concat(system_fits, axis=1)
            system_concat.index.name = "Time (min)"
            system_concat.to_excel(writer, sheet_name=f"{sheet_prefix}_knockouts", index=True)

            # 11. Save Sensitivity Analysis
            if res["perturbation_analysis"] is not None:
                sens_res = res["perturbation_analysis"]
                sens_df = pd.DataFrame(sens_res)
                sens_df.to_excel(writer, sheet_name=f"{sheet_prefix}_sensitivity", index=False)

            # 12. Save Perturbation Results
            if res["perturbation_curves_params"] is not None:
                pert_data = res["perturbation_curves_params"]
                if isinstance(pert_data, pd.DataFrame):
                    pert_df = pert_data.copy()
                else:
                    pert_df = pd.DataFrame(pert_data)
                param_names = sens_df['names'].tolist()
                params_expanded = pd.DataFrame(pert_df["params"].tolist(), columns=param_names)
                sol_array = np.array(pert_df["solution"].tolist())
                n_time, n_states = sol_array.shape[1], sol_array.shape[2]
                time_labels = list(TIME_POINTS) * n_states
                state_labels_repeated = [label for label in state_labels for _ in TIME_POINTS]
                multi_columns = pd.MultiIndex.from_arrays(
                    [time_labels, state_labels_repeated],
                    names=["Time(min)", "State"]
                )
                sol_flat = sol_array.reshape(len(pert_df), -1)
                sol_expanded = pd.DataFrame(sol_flat, columns=multi_columns)
                combined_df = pd.concat([params_expanded, sol_expanded], axis=1)
                combined_df["RMSE"] = pert_df["rmse"]
                combined_df = combined_df.sort_values(by="RMSE").reset_index(drop=True)
                combined_df.to_excel(writer, sheet_name=f"{sheet_prefix}_perturbations", index=False)


def create_report(results_dir: str, output_file: str = f"{model_type}_report.html"):
    """
    Creates a single global report HTML file from all gene folders inside the results directory.

    Args:
        results_dir (str): Path to the root result's directory.
        output_file (str): Name of the generated global report file (placed inside results_dir).
    """
    gene_folders = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    # Build HTML content with updated CSS for spacing.
    html_parts = [
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>Estimation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; font-size: 1.8em; border-bottom: 1px solid #ccc; padding-bottom: 5px; page-break-before: always; }",
        "h2:first-of-type { page-break-before: avoid; }",
        "h3 { color: #666; font-size: 1.4em; margin-top: 10px; margin-bottom: 10px; }",
        # /* CSS grid for plots: two per row, fixed size 500px x 500px, extra space between rows */
        ".plot-container {",
        "  display: grid;",
        "  grid-template-columns: repeat(4, 500px);",
        "  column-gap: 20px;",
        "  row-gap: 40px;",  # // extra vertical gap
        "  justify-content: left;",
        "  margin-bottom: 40px;",
        "}",
        ".plot-item {",
        "  width: 500px;",
        "  height: 500px;",
        "}",
        "img, iframe {",
        "  width: 100%;",
        "  height: 100%;",
        "  object-fit: contain;",
        "  border: none;",
        "}",
        # /* Data tables: full width, one per row */
        ".data-table {",
        "  width: 50%;",
        "  margin-bottom: 20px;",
        "}",
        "table {",
        "  border-collapse: collapse;",
        "  width: 100%;",
        "  margin-top: 10px;",
        "}",
        "th, td {",
        "  border: 1px solid #ccc;",
        "  padding: 8px;",
        "  text-align: left;",
        "}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{model_type.upper()} Model - Parameter Estimation Report</h1>"
    ]
    html_parts += [
        "<pre style=\"font-size: 0.9em; color: #444; background-color: #f9f9f9; padding: 10px; border-left: 4px solid #ccc;\">",
        "A = production of mRNA | B = degradation of mRNA | ",
        "C = production of protein | D = degradation of protein\n",
        "S1, S2, ... = phosphorylation at 1st, 2nd, ... residue | ",
        "D1, D2, ... = degradation of phosphorylated protein at 1st, 2nd, ... residue\n",
        "Sx/Dx (x > 1) = phosphorylation/degradation of intermediate complex at x-th residue",
        "</pre>"
    ]

    # For each gene folder, create a section in the report.
    for gene in sorted(gene_folders):
        gene_folder = os.path.join(results_dir, gene)
        html_parts.append(f"<h2>{gene}</h2>")

        # Create grid container for fixed-size plots.
        html_parts.append('<div class="plot-container">')
        files = sorted(os.listdir(gene_folder))

        for filename in files:
            file_path = os.path.join(gene_folder, filename)

            # PNG plots
            if os.path.isfile(file_path) and filename.endswith(".png"):
                uri_path = Path(file_path).resolve().as_uri()
                base_name = os.path.splitext(filename)[0]
                tokens = [token for token in base_name.split('_') if token]
                if tokens and tokens[0].upper() == gene.upper():
                    tokens = tokens[1:]
                title = " ".join(tokens).upper()
                html_parts.append(
                    f'<div class="plot-item"><h3>{title}</h3><img src="{uri_path}" alt="{filename}"></div>'
                )

        html_parts.append('</div>')  # End of plot container

        # Process data tables and logs in a second loop
        for filename in files:
            file_path = os.path.join(gene_folder, filename)

            # LOG files
            if os.path.isfile(file_path) and filename.endswith(".log"):
                try:
                    with open(file_path, "r", encoding="utf-8") as log_file:
                        log_content = log_file.read()
                    html_parts.append(f'<div class="data-table"><h3>Log File: {filename}</h3>')
                    html_parts.append(
                        '<pre style="font-size: 0.8em; background-color: #f4f4f4; border: 1px solid #ccc; padding: 10px; overflow-x: auto;">'
                    )
                    html_parts.append(log_content)
                    html_parts.append('</pre></div>')
                except Exception as e:
                    html_parts.append(
                        f'<div class="data-table"><h3>Log File: {filename}</h3><p>Error reading log file: {e}</p></div>'
                    )

    html_parts.append("</body>")
    html_parts.append("</html>")

    # Write the report into the result's directory.
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def organize_output_files(directories: Iterable[Union[str, Path]]):
    """
    Organize output files into protein-specific folders and a general folder.

    Args:
        directories (Iterable[Union[str, Path]]): List of directories to organize.
    """
    protein_regex = re.compile(r'([A-Za-z0-9]+)_.*\.(json|svg|png|html|csv|xlsx|tex)$')

    for directory in map(Path, directories):
        if not directory.is_dir():
            print(f"Warning: '{directory}' is not a valid directory. Skipping.")
            continue

        # Move files matching the protein pattern
        for file_path in directory.iterdir():
            if file_path.is_file():
                match = protein_regex.search(file_path.name)
                if match:
                    protein = match.group(1)
                    protein_folder = directory / protein
                    protein_folder.mkdir(exist_ok=True)
                    shutil.move(str(file_path), str(protein_folder / file_path.name))

        # Identify remaining files
        remaining_files = [f for f in directory.iterdir() if f.is_file()]
        if remaining_files:
            general_folder = directory / "General"
            general_folder.mkdir(exist_ok=True)
            for file_path in remaining_files:
                shutil.move(str(file_path), str(general_folder / file_path.name))