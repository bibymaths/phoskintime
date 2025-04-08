import os, re, shutil
import pandas as pd
from tfopt.evol.config.constants import INPUT3, INPUT1, INPUT4

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def load_mRNA_data(filename=INPUT3):
    df = pd.read_csv(filename)
    mRNA_ids = df["GeneID"].astype(str).tolist()
    time_cols = [col for col in df.columns if col != "GeneID"]
    mRNA_mat = df[time_cols].to_numpy(dtype=float)
    return mRNA_ids, mRNA_mat, time_cols

def load_TF_data(filename=INPUT1):
    df = pd.read_csv(filename)
    protein_dict = {}
    psite_dict = {}
    psite_labels_dict = {}
    for _, row in df.iterrows():
        tf = str(row["GeneID"]).strip()
        psite = str(row["Psite"]).strip()
        time_cols = [col for col in df.columns if col not in ["GeneID", "Psite"]]
        vals = row[time_cols].to_numpy(dtype=float)
        if tf not in protein_dict:
            protein_dict[tf] = None
            psite_dict[tf] = []
            psite_labels_dict[tf] = []
        if psite == "" or psite.lower() == "nan" or psite.startswith('A_') or psite.startswith('A_'):
            protein_dict[tf] = vals
        else:
            psite_dict[tf].append(vals)
            psite_labels_dict[tf].append(psite)
    TF_ids = list(protein_dict.keys())
    return TF_ids, protein_dict, psite_dict, psite_labels_dict, time_cols

def load_regulation(filename=INPUT4):
    """
    Assumes the regulation file is reversed:
      - The 'Source' column holds mRNA identifiers.
      - The 'Target' column holds TF identifiers.
    Returns a mapping from mRNA (source) to a list of TFs (targets).
    """
    df = pd.read_csv(filename)
    reg_map = {}
    for _, row in df.iterrows():
        mrna = str(row["Source"]).strip()
        tf = str(row["Target"]).strip()
        if mrna not in reg_map:
            reg_map[mrna] = []
        if tf not in reg_map[mrna]:
            reg_map[mrna].append(tf)
    return reg_map

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
        "  row-gap: 40px;", # /* extra vertical gap */
        "  justify-content: left;",
        "  margin-bottom: 20px;",
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
        "<h1>Global Report</h1>"
    ]

    # For each gene folder, create a section in the report.
    for gene in sorted(gene_folders):
        gene_folder = os.path.join(results_dir, gene)
        html_parts.append(f"<h2>mRNA: {gene}</h2>")

        # Create grid container for fixed-size plots.
        html_parts.append('<div class="plot-container">')
        files = sorted(os.listdir(gene_folder))
        for filename in files:
            file_path = os.path.join(gene_folder, filename)
            if os.path.isfile(file_path):
                if filename.endswith(".png"):
                    rel_path = os.path.join(gene, filename)
                    html_parts.append(
                        f'<div class="plot-item"><h3>{filename}</h3><img src="{rel_path}" alt="{filename}"></div>'
                    )
        html_parts.append('</div>')  # End of plot container

        # Data tables: display XLSX or CSV files from the gene folder, one per row.
        for filename in files:
            file_path = os.path.join(gene_folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".xlsx"):
                try:
                    df = pd.read_excel(file_path)
                    table_html = df.to_html(index=False, border=0)
                    html_parts.append(f'<div class="data-table"><h3>Data Table: {filename}</h3>{table_html}</div>')
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

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"