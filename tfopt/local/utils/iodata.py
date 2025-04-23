import os, re, shutil
import pandas as pd
import numpy as np
from tfopt.local.config.constants import INPUT3, INPUT1, INPUT4


def min_max_normalize(df, custom_max=None):
    """
    Row-wise (per-sample) min-max normalize time-series columns starting with 'x'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with time-series columns (x1-xN).
        custom_max (float, optional): If given, used as max for all rows.

    Returns:
        pd.DataFrame: Normalized DataFrame with same shape.
    """
    df = df.copy()
    time_cols = [col for col in df.columns if col.startswith("x")]
    data = df[time_cols].to_numpy(dtype=float)

    row_min = np.min(data, axis=1, keepdims=True)
    if custom_max is not None:
        denom = (custom_max - row_min)
    else:
        row_max = np.max(data, axis=1, keepdims=True)
        denom = (row_max - row_min)

    normalized = (data - row_min) / denom
    df[time_cols] = normalized

    return df

def load_expression_data(filename=INPUT3):
    """
    Loads gene expression (mRNA) data.
    Expects a CSV with a 'GeneID' column and time-point columns.
    """
    df = pd.read_csv(filename)
    # Normalize for high unscaled variability
    # Exists often
    # df = min_max_normalize(df)
    gene_ids = df["GeneID"].astype(str).tolist()
    time_cols = [col for col in df.columns if col != "GeneID"]
    expression_matrix = df[time_cols].to_numpy(dtype=float)
    return gene_ids, expression_matrix, time_cols


def load_tf_protein_data(filename=INPUT1):
    """
    Loads TF protein data along with PSite information.
    Expects a CSV with 'GeneID' and 'Psite' columns.
    For rows without a valid PSite, the entire row is considered as the protein signal.
    """
    expr_gene_ids, expression_matrix, expr_time_cols = load_expression_data()
    df = pd.read_csv(filename)
    # Normalize for high unscaled variability
    # Exists often
    # df = min_max_normalize(df)
    tf_protein = {}
    tf_psite_data = {}
    tf_psite_labels = {}
    # Original time columns from TF data (should be 14 columns)
    orig_time_cols = [col for col in df.columns if col not in ["GeneID", "Psite"]]
    # Use only time points from index 5 onward (i.e. last 9 time points) if available.
    # To match the expression data, we need to ensure the same number of time points.
    if len(orig_time_cols) >= 14:
        time_cols = orig_time_cols[5:]
    else:
        time_cols = orig_time_cols
    for _, row in df.iterrows():
        tf = str(row["GeneID"]).strip()
        psite = str(row["Psite"]).strip()
        if not psite.startswith(("S_", "Y_", "T_")):
            continue  # Skip psite values that don't start with S_, Y_, or T_
        vals = row[orig_time_cols].to_numpy(dtype=float)
        vals = vals[5:] if len(orig_time_cols) >= 14 else vals
        if tf not in tf_protein:
            tf_protein[tf] = vals
            tf_psite_data[tf] = []
            tf_psite_labels[tf] = []
        else:
            tf_psite_data[tf].append(vals)
            tf_psite_labels[tf].append(psite)

    # Add expression data to tf_protein (only if not already present)
    for gene_id, expr_vals in zip(expr_gene_ids, expression_matrix):
        if gene_id not in tf_protein:
            tf_protein[gene_id] = expr_vals
            tf_psite_data[gene_id] = []
            tf_psite_labels[gene_id] = []

    tf_ids = list(tf_protein.keys())

    return tf_ids, tf_protein, tf_psite_data, tf_psite_labels, time_cols

def load_regulation(filename=INPUT4):
    """
    Assumes the regulation file is reversed:
      - The 'Source' column holds gene (mRNA) identifiers.
      - The 'Target' column holds TF identifiers.
    Returns a mapping from gene (source) to a list of TFs (targets).
    """
    df = pd.read_csv(filename)
    reg_map = {}
    for _, row in df.iterrows():
        tf = str(row["Source"]).strip()
        gene = str(row["Target"]).strip()
        if gene not in reg_map:
            reg_map[gene] = []
        if tf not in reg_map[gene]:
            reg_map[gene].append(tf)
    return reg_map

def summarize_stats(input3=INPUT3, input1=INPUT1, input4=INPUT4):
    """
    Summarizes statistics for the expression data (input3) and TF protein data (input1).
    It also summarizes the data after filtering based on the mapping file (input4).

    The function prints the following statistics:
        - Global min, max, std, var for the full dataset.
        - Time-wise min, max, std, var for each time point.
        - Global min, max, std, var for the subset data (filtered by input4).
        - Time-wise min, max, std, var for the subset data.
    Args:
        input3 (str): Path to the expression data CSV file.
        input1 (str): Path to the TF protein data CSV file.
        input4 (str): Path to the mapping file CSV.
    """
    # Load input3: expression data
    expr_df = pd.read_csv(input3)
    expr_data = expr_df.drop(columns=["GeneID"])

    print("=== Expression Data (input3) — Full Dataset ===")
    print(f"Global min: {expr_data.values.min():.4f}")
    print(f"Global max: {expr_data.values.max():.4f}")
    print(f"Global std: {expr_data.values.std():.4f}")
    print(f"Global var: {expr_data.values.var():.4f}")
    print("\nTime-wise stats:")
    print(expr_data.agg(['min', 'max', 'std', 'var']).T)

    # Load input1: TF protein data
    prot_df = pd.read_csv(input1)
    time_cols = [col for col in prot_df.columns if col not in ["GeneID", "Psite"]]
    prot_data = prot_df[time_cols]

    print("\n=== TF Protein Data (input1) — Full Dataset ===")
    print(f"Global min: {prot_data.values.min():.4f}")
    print(f"Global max: {prot_data.values.max():.4f}")
    print(f"Global std: {prot_data.values.std():.4f}")
    print(f"Global var: {prot_data.values.var():.4f}")
    print("\nTime-wise stats:")
    print(prot_data.agg(['min', 'max', 'std', 'var']).T)

    # Load mapping from input4
    map_df = pd.read_csv(input4)
    expr_subset = expr_df[expr_df["GeneID"].isin(map_df["Source"])]
    prot_subset = prot_df[prot_df["GeneID"].isin(map_df["Target"])]

    print("\n=== Expression Data — Subset from input4 ===")
    expr_data_sub = expr_subset.drop(columns=["GeneID"])
    print(f"Global min: {expr_data_sub.values.min():.4f}")
    print(f"Global max: {expr_data_sub.values.max():.4f}")
    print(f"Global std: {expr_data_sub.values.std():.4f}")
    print(f"Global var: {expr_data_sub.values.var():.4f}")
    print("\nTime-wise stats:")
    print(expr_data_sub.agg(['min', 'max', 'std', 'var']).T)

    print("\n=== TF Protein Data — Subset from input4 ===")
    prot_data_sub = prot_subset[time_cols]
    print(f"Global min: {prot_data_sub.values.min():.4f}")
    print(f"Global max: {prot_data_sub.values.max():.4f}")
    print(f"Global std: {prot_data_sub.values.std():.4f}")
    print(f"Global var: {prot_data_sub.values.var():.4f}")
    print("\nTime-wise stats:")
    print(prot_data_sub.agg(['min', 'max', 'std', 'var']).T)

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
        "<h1>[Local] mRNA-TF Optimization Report</h1>"
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