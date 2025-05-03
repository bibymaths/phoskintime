import os
from os.path import exists

import pandas as pd
from pathlib import Path
from config.logconf import setup_logger

logger = setup_logger()

ROOT = Path(__file__).resolve().parent.parent  # …/phoskintime
BASE = Path(__file__).parent  # …/processing
MAPPING = ROOT / "mapping"
os.makedirs(MAPPING, exist_ok=True)

def map_optimization_results(tf_file_path, kin_file_path, sheet_name='Alpha Values'):
    """
    Reads the TF-mRNA optimization results from an Excel file and maps mRNA to each TF.

    This function processes the 'Alpha Values' sheet of the provided Excel file to extract
    non-zero optimization results, groups mRNA by TF, and merges the results with additional
    data from a CSV file containing TF, Psite, and Kinase information. The final DataFrame
    is cleaned and formatted for further analysis.

    Args:
        tf_file_path: Path to the Excel file containing TF-mRNA optimization results.
        kin_file_path: Path to the Excel file containing Kinase-Phosphorylation optimization results.
        sheet_name: The name of the sheet in the Excel file to read from. Default is 'Alpha Values'.

    Returns:
        pd.DataFrame: A DataFrame containing the mapped TF, mRNA, Psite, and Kinase information.
    """
    # --- TF mapping ---
    tfopt_file = pd.ExcelFile(tf_file_path)
    df = pd.read_excel(tfopt_file, sheet_name=sheet_name)
    non_zero_df = df[df['Value'] > 1e-3]

    # Group TFs and their corresponding strengths for each mRNA
    tf_grouped = non_zero_df.groupby('mRNA').agg({
        'TF': lambda x: ', '.join(x.astype(str)),
        'Value': lambda x: ', '.join(map(str, x))
    }).reset_index().rename(columns={'Value': 'TF_strength'})

    # --- Kinase mapping ---
    kin_xls = pd.ExcelFile(kin_file_path)
    kin_df = pd.read_excel(kin_xls, sheet_name=sheet_name)
    kin_df = kin_df[kin_df['Alpha'] > 1e-3]

    # Rename for consistency
    kin_df = kin_df.rename(columns={'Gene': 'mRNA'})

    kin_grouped = kin_df.groupby(['mRNA', 'Psite']).agg({
        'Kinase': lambda x: ', '.join(x.astype(str)),
        'Alpha': lambda x: ', '.join(map(str, x))
    }).reset_index().rename(columns={'Alpha': 'Kinase_strength'})

    # --- Merge results ---
    merged_df = pd.merge(kin_grouped, tf_grouped, on='mRNA', how='left')

    # Hide repeated TF rows for readability
    merged_df['TF'] = merged_df['TF'].where(merged_df['TF'] != merged_df['TF'].shift(), '')
    merged_df['TF_strength'] = merged_df['TF_strength'].where(merged_df['TF_strength'] != merged_df['TF_strength'].shift(), '')

    return merged_df


def create_cytoscape_table(mapping_csv_path):
    """
    Creates a Cytoscape-compatible edge table from a mapping file.
    Adds a 'Strength' column from TF_strength or Kinase_strength.

    Parameters:
        mapping_csv_path (str): Path to the input CSV file with columns:
                                TF, TF_strength, mRNA, Psite, Kinase, Kinase_strength

    Returns:
        pd.DataFrame: Edge table with columns [Source, Target, Interaction, Strength]
    """
    df = pd.read_csv(mapping_csv_path)

    kinase_tf_edges = []
    tf_mrna_edges = []

    for _, row in df.iterrows():
        mRNA = row["mRNA"]
        psite = row.get("Psite", "")

        # Kinase -> mRNA
        if pd.notna(row["Kinase"]):
            kin_strengths = str(row.get("Kinase_strength", "")).split(",")
            kinases = str(row["Kinase"]).split(",")
            for i, kinase in enumerate(kinases):
                strength = kin_strengths[i].strip() if i < len(kin_strengths) else ""
                kinase_tf_edges.append({
                    "Source": kinase.strip(),
                    "Target": mRNA.strip(),
                    "Interaction": "phosphorylates",
                    "Strength": strength,
                    "Psite": psite
                })

        # TF -> mRNA
        if pd.notna(row["TF"]):
            tf_strengths = str(row.get("TF_strength", "")).split(",")
            tfs = str(row["TF"]).split(",")
            for i, tf in enumerate(tfs):
                strength = tf_strengths[i].strip() if i < len(tf_strengths) else ""
                tf_mrna_edges.append({
                    "Source": tf.strip(),
                    "Target": mRNA.strip(),
                    "Interaction": "regulates",
                    "Strength": strength,
                    "Psite": ""
                })

    edge_df = pd.DataFrame(kinase_tf_edges + tf_mrna_edges)
    return edge_df

def add_kinetic_strength_columns(mapping_path, mapping__path, excel_path, suffix):
    """
    Adds kinetic strength columns to the mapping files based on the provided Excel file.
    The function reads the mapping files and the Excel file, extracts the kinetic strength values,
    and updates the mapping files with the new columns.
    The updated mapping files are saved with a specified suffix of the model.
    """
    mapping = pd.read_csv(mapping_path)
    mapping_ = pd.read_csv(mapping__path)
    all_sheets = pd.read_excel(excel_path, sheet_name=None)

    def build_psite_order(df, gene_col):
        return df.groupby(gene_col)["Psite"].apply(lambda x: list(dict.fromkeys(x))).to_dict()

    def get_strength(gene, psite, psite_order):
        sheet = all_sheets.get(f"{gene}_params")
        if sheet is None:
            return None
        try:
            idx = psite_order[gene].index(psite) + 1
            return sheet.at[0, f"S{idx}"] if f"S{idx}" in sheet.columns else None
        except Exception:
            return None

    def enrich(df, gene_col, psite_order):
        df = df.copy()
        df["Kinetic_strength"] = df.apply(lambda row: get_strength(row[gene_col], row["Psite"], psite_order), axis=1)
        return df

    order_map = build_psite_order(mapping, "mRNA")
    order_map_ = build_psite_order(mapping_, "Target")

    updated = enrich(mapping, "mRNA", order_map)
    updated_ = enrich(mapping_, "Target", order_map_)

    updated.to_csv(ROOT / MAPPING / f"mapping_{suffix}.csv", index=False)
    updated_.to_csv(ROOT / MAPPING / f"mapping_{suffix}_.csv", index=False)

def generate_nodes(edge_df):
    """
    Infers node types and aggregates all Psites per target node from phosphorylation edges.

    Parameters:
        edge_df (pd.DataFrame): Must have columns ['Source', 'Target', 'Interaction', 'Psite']

    Returns:
        pd.DataFrame: DataFrame with columns ['Node', 'Type', 'Psite']
    """
    node_info = {}

    for _, row in edge_df.iterrows():
        src, tgt, interaction = row["Source"], row["Target"], row["Interaction"]
        psite = row.get("Psite", "")

        # Source
        if interaction == "regulates":
            node_info.setdefault(src, {"Type": "TF", "Psite_set": set()})
            node_info.setdefault(tgt, {"Type": "Kinase", "Psite_set": set()})
        else:  # phosphorylates
            node_info.setdefault(src, {"Type": "Kinase", "Psite_set": set()})
            node_info.setdefault(tgt, {"Type": "Kinase", "Psite_set": set()})
            if psite:
                node_info[tgt]["Psite_set"].add(str(psite).strip())

    node_records = []
    for node, info in node_info.items():
        psite_list = sorted(info["Psite_set"])  # consistent order
        node_records.append({
            "Node": node,
            "Type": info["Type"],
            "Psite": ", ".join(psite_list) if psite_list else ""
        })

    return pd.DataFrame(node_records)


if __name__ == "__main__":

    # Path to the Excel file of mRNA-TF optimization results
    tf_file_path = ROOT / "data" / "tfopt_results.xlsx"
    kin_file_path = ROOT / "data" / "kinopt_results.xlsx"

    # Call the function to map optimization results
    mapped_df = map_optimization_results(tf_file_path, kin_file_path)

    # Save the mapped DataFrame to a CSV file
    mapped_df.to_csv('mapped_TF_mRNA_phospho.csv', index=False)

    # Create a Cytoscape-compatible edge table
    edge_table = create_cytoscape_table('mapped_TF_mRNA_phospho.csv')

    # Save the edge table to a CSV file
    edge_table.to_csv('mapping_table.csv', index=False)

    # Generate nodes for Cytoscape
    nodes_df = generate_nodes(edge_table)

    # Save the nodes DataFrame to a CSV file
    nodes_df.to_csv('nodes.csv', index=False)

    # Move the files to the data folder
    os.rename('nodes.csv', ROOT / MAPPING / 'nodes.csv')
    os.rename('mapped_TF_mRNA_phospho.csv', ROOT / MAPPING / 'mapping.csv')
    os.rename('mapping_table.csv', ROOT / MAPPING / 'mapping_.csv')

    for mod in ["Distributive", "Successive", "Random"]:
        mod_path = ROOT / f"{mod}_results" / mod / f"{mod}_results.xlsx"
        if mod_path.exists():
            add_kinetic_strength_columns(
                mapping_path=ROOT / MAPPING / 'mapping.csv',
                mapping__path=ROOT / MAPPING / 'mapping_.csv',
                excel_path=mod_path,
                suffix=mod
            )

    logger.info(f"Mapping files for merging with ODE results & further use in Cytoscape")

    for suffix in ["Distributive", "Successive", "Random"]:
        for fname in [f"mapping_{suffix}.csv", f"mapping_{suffix}_.csv"]:
            fpath = ROOT / MAPPING / fname
            if fpath.exists():
                logger.info(f"{fpath.as_uri()}")

    for fpath in [ROOT / MAPPING / 'nodes.csv',
                  ROOT / MAPPING / 'mapping.csv',
                  ROOT / MAPPING / 'mapping_.csv']:
        logger.info(f"{fpath.as_uri()}")

# Note: The following comment is an example of how to handle missing data
# PAK2 is not included in the mapping file because it doesn't exist in the input4.csv file.

""" 
PAK2 doesn't exist in CollectTRI (so also not in input4.csv) but it is  
in the GeneID column of the phospho interaction file. 
No mRNA-TF estimation is possible for PAK2.  
Hence, it is not included in the output mapped file for ODE modelling.
"""
