import os, re, shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from kinopt.evol.config.constants import INPUT1, INPUT2

def format_duration(seconds):
    """
    Returns a formatted string representing the duration in seconds, minutes, or hours.
    The format is:
    - seconds: "xx.xx sec"
    - minutes: "xx.xx min"
    - hours: "xx.xx hr"

    :param seconds:
    :return:
    """
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} hr"

def load_and_scale_data(estimate_missing, scaling_method, split_point, seg_points):
    """
    Loads the full HGNC data and kinase interaction data, applies scaling to the time-series columns,
    and subsets/merges them. The first file is the full HGNC data, and the second file contains kinase interactions.
    The function also handles the conversion of kinases from string format to list format.

    :param estimate_missing:
    :param scaling_method:
    :param split_point:
    :param seg_points:

    :return:
    full_hgnc_df (pd.DataFrame): The scaled data from input1
    interaction_df (pd.DataFrame): The subset/merged DataFrame from input2
    observed (pd.DataFrame): Subset of full_hgnc_df merged with interaction_df
    """
    full_hgnc_df = pd.read_csv(INPUT1)
    time_series_cols = [f'x{i}' for i in range(1, 15)]
    full_hgnc_df = apply_scaling(full_hgnc_df, time_series_cols, scaling_method, split_point, seg_points)
    interaction_df = pd.read_csv(INPUT2, header=0)
    if estimate_missing:
        observed = full_hgnc_df.merge(interaction_df.iloc[:, :2], on=["GeneID", "Psite"]).drop(columns=["max", "min"])
        interaction_df['Kinase'] = interaction_df['Kinase'].str.strip('{}').apply(lambda x: [k.strip() for k in x.split(',')])
    else:
        interaction_df = interaction_df[interaction_df['Kinase'].apply(
            lambda k: all(kinase in set(full_hgnc_df['GeneID'][1:]) for kinase in k.strip('{}').split(',')))]
        interaction_df['Kinase'] = interaction_df['Kinase'].str.strip('{}').apply(lambda x: [k.strip() for k in x.split(',')])
        observed = full_hgnc_df.merge(interaction_df.iloc[:, :2], on=["GeneID", "Psite"]).drop(columns=["max", "min"])
    return full_hgnc_df, interaction_df, observed

def apply_scaling(df, time_series_columns, method, split_point, segment_points):
    """
    Applies different scaling methods to the time-series columns of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the time-series data.
        time_series_columns (list): List of columns to be scaled.
        method (str): The scaling method to apply ('min_max', 'log', 'temporal', 'segmented', 'slope', 'cumulative').
        split_point (int): Column index for temporal scaling.
        segment_points (list): List of column indices for segmented scaling.
    :returns:
        pd.DataFrame: The DataFrame with scaled time-series columns.
    """

    if method == 'min_max':
        # Min-Max Scaling (row-wise)
        scaler = MinMaxScaler()
        df[time_series_columns] = pd.DataFrame(
            df[time_series_columns]
            .apply(lambda row: scaler.fit_transform(row.values.reshape(-1, 1)).flatten(), axis=1).tolist(),
            index=df.index
        )

    elif method == 'log':
        # Log Scaling
        df[time_series_columns] = df[time_series_columns].applymap(lambda x: np.log(x))

    elif method == 'temporal':
        # Temporal Scaling (first split_point columns vs the rest)
        first_part_columns = time_series_columns[:split_point]
        second_part_columns = time_series_columns[split_point:]

        scaler_first = MinMaxScaler()
        scaler_second = MinMaxScaler()

        df[first_part_columns] = scaler_first.fit_transform(df[first_part_columns])
        df[second_part_columns] = scaler_second.fit_transform(df[second_part_columns])

    elif method == 'segmented':
        # Segmented Scaling (multiple segments)
        if not segment_points:
            raise ValueError("segment_points must be provided for segmented scaling.")
        segment_columns = [time_series_columns[segment_points[i]:segment_points[i + 1]] for i in
                           range(len(segment_points) - 1)]
        for segment in segment_columns:
            scaler = MinMaxScaler()
            df[segment] = scaler.fit_transform(df[segment])

    elif method == 'slope':
        # Slope Scaling (rate of change)
        diffs_df = df[time_series_columns].diff(axis=1).fillna(0)
        scaler = MinMaxScaler()
        df[time_series_columns] = scaler.fit_transform(diffs_df)

    elif method == 'cumulative':
        # Cumulative Scaling (cumulative sum)
        cum_df = df[time_series_columns].cumsum(axis=1)
        scaler = MinMaxScaler()
        df[time_series_columns] = scaler.fit_transform(cum_df)

    elif method == "None":
        return df

    else:
        raise ValueError(
            "Invalid scaling method. Choose from 'min_max', 'log', 'temporal', 'segmented', 'slope', 'cumulative'.")

    return df

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
        "<h1>Kinase Optimization Report</h1>"
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
    """
    Organizes output files from the optimization process into separate folders based on protein names.
    Each protein's files are moved into a folder named after the protein.
    Any remaining files that do not match the protein pattern are moved to a "General" folder.

    :param directories: List of directories to organize.
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