import os, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd

from config.constants import TIME_POINTS, model_type, ESTIMATION_MODE


def ensure_output_directory(directory):
    """
    Ensure the output directory exists. If it doesn't, create it.

    :param directory: str
    """
    os.makedirs(directory, exist_ok=True)

def load_data(excel_file, sheet="Estimated Values"):
    """
    Load data from an Excel file. The default sheet is "Estimated Values".

    :param excel_file: str
    :param sheet: str
    :return: DataFrame
    :rtype: pd.DataFrame
    """
    return pd.read_excel(excel_file, sheet_name=sheet)

def format_duration(seconds):
    """
    Format a duration in seconds into a human-readable string.
    The function converts seconds into a string representation in the format:
    - "X sec" for seconds
    - "X min" for minutes
    - "X hr" for hours

    :param seconds: float
    :return: formatted string
    """
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"

def merge_obs_est(filename):
    """
    Loads observed and estimated data from an Excel file where:
    - Each gene has two sheets: <GENE>_site_observed and <GENE>_site_estimates
    - Rows = Psites with index "Site/Time(min)"
    - Columns = time points (14)

    Returns:
        A DataFrame with columns: Gene, Psite, x1_obs–x14_obs, x1_est–x14_est
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
    Save the results to an Excel file with multiple sheets.
    Each sheet corresponds to a different gene and contains:
    - Sequential Parameter Estimates
    - Profiled Estimates (if available)
    - Errors summary
    - Model fits for the system
    - Model fits for each site
    - Observed data for each site
    - PCA results
    - t-SNE results
    - Knockout results
    The sheet names are prefixed with the gene name, truncated to 25 characters.
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
            ev_row = pd.DataFrame([["Explained Var"] + list(ev)],columns=pca_df.columns)
            pca_df = pd.concat([pca_df, ev_row], ignore_index=True)
            pca_df.to_excel(writer, sheet_name=f"{sheet_prefix}_pca", index=False)

            # 9. Save t-SNE results
            tsne_arr = np.array(res["tsne_result"])
            tsne_df = pd.DataFrame(tsne_arr, columns=[f"t-SNE{i+1}" for i in range(tsne_arr.shape[1])])
            tsne_df.insert(0, "Time(min)", TIME_POINTS)
            tsne_df.to_excel(writer, sheet_name=f"{sheet_prefix}_tsne", index=False)

            # 10. Save knockout results
            if "knockout_results" in res and res["knockout_results"]:
                system_fits = []  # for R and P
                phospho_fits = []  # for phospho sites

                for knockout_name, ko_data in res["knockout_results"].items():
                    sol_ko = np.array(ko_data["sol_ko"])  # (time, 2)
                    p_fit_ko = np.array(ko_data["p_fit_ko"])  # (psites, time)

                    # System dynamics
                    state_labels = ["R", "P"] + [f"" for ps in res["psite_labels"]]
                    sys_df = pd.DataFrame(sol_ko, columns=[f"{knockout_name} {label}" for label in state_labels],
                                          index=TIME_POINTS)
                    system_fits.append(sys_df)

                # -- Concatenate system dynamics
                system_concat = pd.concat(system_fits, axis=1)
                system_concat.index.name = "Time (min)"
                system_concat.to_excel(writer, sheet_name=f"{sheet_prefix}_knockouts_system", index=True)

def create_report(results_dir: str, output_file: str = "report.html"):
    """
    Creates a single global report HTML file from all gene folders inside the results directory.

    For each gene folder (e.g. "ABL2"), the report will include:
      - All PNG plots and interactive HTML plots displayed in a grid with three plots per row.
      - Each plot is confined to a fixed size of 900px by 900px.
      - Data tables from XLSX or CSV files in the gene folder are displayed below the plots, one per row.

    Args:
        results_dir (str): Path to the root result's directory.
        output_file (str): Name of the generated global report file (placed inside results_dir).
    """
    # Gather gene folders (skip "General" and "logs")
    gene_folders = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d not in ("General", "logs")
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
        "  grid-template-columns: repeat(2, 500px);",
        "  column-gap: 20px;",
        "  row-gap: 40px;",  #// extra vertical gap
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
        f"<h1>{model_type.upper()} Modelling & {ESTIMATION_MODE.upper()} Parameter Estimation Report</h1>"
    ]

    # For each gene folder, create a section in the report.
    for gene in sorted(gene_folders):
        gene_folder = os.path.join(results_dir, gene)
        html_parts.append(f"<h2>Protein Group: {gene}</h2>")

        # Create grid container for fixed-size plots.
        html_parts.append('<div class="plot-container">')
        files = sorted(os.listdir(gene_folder))
        for filename in files:
            file_path = os.path.join(gene_folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".png"):
                uri_path = Path(os.path.join(gene_folder, filename)).resolve().as_uri()
                # Remove the extension and split on '_'
                base_name = os.path.splitext(filename)[0]
                tokens = [token for token in base_name.split('_') if token]
                # Remove the gene name if it matches (case-insensitive)
                if tokens and tokens[0].upper() == gene.upper():
                    tokens = tokens[1:]
                # Join remaining tokens with space and convert to upper case
                title = " ".join(tokens).upper()
                html_parts.append(
                    f'<div class="plot-item"><h3>{title}</h3><img src="{uri_path}" alt="{filename}"></div>'
                )
        html_parts.append('</div>')  # End of plot container

        # Data tables: display XLSX or CSV files from the gene folder, one per row.
        for filename in files:
            file_path = os.path.join(gene_folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".xlsx"):
                try:
                    df = pd.read_excel(file_path)
                    table_html = df.to_html(index=False, border=0)
                    # Remove the extension and split on '_'
                    base_name = os.path.splitext(filename)[0]
                    tokens = [token for token in base_name.split('_') if token]
                    # Remove the gene name if it matches (case-insensitive)
                    if tokens and tokens[0].upper() == gene.upper():
                        tokens = tokens[1:]
                    # Join remaining tokens with space and convert to upper case
                    title = " ".join(tokens).upper()
                    html_parts.append(f'<div class="data-table"><h3>Data Table: {title}</h3>{table_html}</div>')
                except Exception as e:
                    html_parts.append(
                        f'<div class="data-table"><h3>Data Table: {filename}</h3><p>Error reading {filename}: {e}</p></div>'
                    )

    html_parts.append("</body>")
    html_parts.append("</html>")

    # Write the report into the results directory.
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def organize_output_files(*directories):
    """
    Organize output files into protein-specific folders and a general folder.
    Files matching the pattern "protein_name_*.{json,svg,png,html,csv,xlsx}"
    will be moved to a folder named after the protein.
    Remaining files will be moved to a "General" folder within the same directory.

    :param directories: List of directories to organize.
    :type directories: list
    """
    protein_regex = re.compile(r'([A-Za-z0-9]+)_.*\.(json|svg|png|html|csv|xlsx)$')

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
