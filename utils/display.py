import os, re, shutil
import pandas as pd

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

def save_result(results, excel_filename):
    """
    Save the results to an Excel file with multiple sheets.
    Each sheet corresponds to a different gene and contains:
    - Sequential Parameter Estimates
    - Profiled Estimates (if available)
    - Errors summary
    The sheet names are prefixed with the gene name, truncated to 25 characters.
    Args:
        results (list): List of dictionaries containing results for each gene.
        excel_filename (str): Path to the output Excel file.
    """
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for res in results:
            gene = res["gene"]
            sheet_prefix = gene[:25]  # Excel sheet names must be ≤31 chars

            # 1. Save Sequential Parameter Estimates
            param_df = res["param_df"].copy()
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

def create_report(results_dir: str, output_file: str = "report.html"):
    """
    Creates a single global report HTML file from all gene folders inside the results directory.

    For each gene folder (e.g. "ABL2"), the report will include:
      - All PNG plots and interactive HTML plots displayed in a grid with three plots per row.
      - Each plot is confined to a fixed size of 900px by 900px.
      - Data tables from XLSX or CSV files in the gene folder are displayed below the plots, one per row.

    Args:
        results_dir (str): Path to the root results directory.
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
        "h2 { color: #555; font-size: 1.8em; border-bottom: 1px solid #ccc; padding-bottom: 5px; }",
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
        "<h1>Modelling & Parameter Estimation Report</h1>"
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
                rel_path = os.path.join(gene, filename)
                # Remove the extension and split on '_'
                base_name = os.path.splitext(filename)[0]
                tokens = [token for token in base_name.split('_') if token]
                # Remove the gene name if it matches (case-insensitive)
                if tokens and tokens[0].upper() == gene.upper():
                    tokens = tokens[1:]
                # Join remaining tokens with space and convert to upper case
                title = " ".join(tokens).upper()
                html_parts.append(
                    f'<div class="plot-item"><h3>{title}</h3><img src="{rel_path}" alt="{filename}"></div>'
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
