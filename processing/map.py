import os
import pandas as pd
from pathlib import Path
from config.logconf import setup_logger

logger = setup_logger()

ROOT = Path(__file__).resolve().parent.parent  # …/phoskintime
BASE = Path(__file__).parent  # …/processing


def map_optimization_results(file_path):
    """
    Reads the TF-mRNA optimization results from an Excel file and maps mRNA to each TF.

    This function processes the 'Alpha Values' sheet of the provided Excel file to extract
    non-zero optimization results, groups mRNA by TF, and merges the results with additional
    data from a CSV file containing TF, Psite, and Kinase information. The final DataFrame
    is cleaned and formatted for further analysis.

    Args:
        file_path (str): The path to the Excel file containing TF-mRNA optimization results.

    Returns:
        pd.DataFrame: A DataFrame containing the mapped TF, mRNA, Psite, and Kinase information.
    """
    # Read the Excel file
    tfopt_file = pd.ExcelFile(file_path)

    # Read the 'Alpha Values' sheet
    df = pd.read_excel(tfopt_file, sheet_name='Alpha Values')

    # Filter the DataFrame for non-zero values
    non_zero_df = df[df['Value'] != 0]

    # Extract the mRNA for each TF where Values was not zero
    result = non_zero_df[['mRNA', 'TF']]

    result = result.groupby('mRNA').agg(lambda x: ', '.join(x)).reset_index()

    # Read another csv file which has TF aka GeneID and Psites and Kinases to merge with the result
    kinopt_file = pd.read_csv(BASE / "raw" / "input2.csv")
    df2 = kinopt_file.rename(columns={'GeneID': 'mRNA'})
    merged_df = pd.merge(result, df2, on='mRNA', how='left')

    # Remove {} from the Kinases column with format {kinase1, kinase2, kinase3}
    merged_df['Kinase'] = merged_df['Kinase'].astype(str)
    merged_df['Kinase'] = merged_df['Kinase'].str.replace(r'{', '', regex=True)
    merged_df['Kinase'] = merged_df['Kinase'].str.replace(r'}', '', regex=True)

    # Remove any extra spaces
    merged_df['Kinase'] = merged_df['Kinase'].str.replace(r'\s+', '', regex=True)

    # Delete the rows where Psite doesn't match Psite in input2
    merged_df = merged_df[merged_df['Psite'].isin(df2['Psite'])]

    # Empty cells in mRNA column if they are repeateed in the next row
    merged_df['TF'] = merged_df['TF'].where(merged_df['TF'] != merged_df['TF'].shift(), '')

    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.reset_index(drop=True)

    return merged_df


def create_cytoscape_table(mapping_csv_path):
    """
    Creates a Cytoscape-compatible edge table from a mapping file.

    Parameters:
        mapping_csv_path (str): Path to the input CSV file with columns: TF, mRNA, Psite, Kinase

    Returns:
        pd.DataFrame: Edge table with columns [Source, Target, Interaction]
    """
    df = pd.read_csv(mapping_csv_path)

    kinase_tf_edges = []
    tf_mrna_edges = []

    for _, row in df.iterrows():
        mRNA = row["mRNA"]

        # Add Kinase -> mRNA edges
        if pd.notna(row["Kinase"]):
            for kinase in str(row["Kinase"]).split(","):
                kinase_tf_edges.append((kinase.strip(), mRNA.strip(), "phosphorylates"))

        # Add TF -> mRNA edges
        if pd.notna(row["TF"]):
            for tf in str(row["TF"]).split(","):
                tf_mrna_edges.append((tf.strip(), mRNA.strip(), "regulates"))

    edge_df = pd.DataFrame(kinase_tf_edges + tf_mrna_edges,
                           columns=["Source", "Target", "Interaction"])
    return edge_df


def generate_nodes(edge_df):
    """
    Infers node types for Cytoscape visualization:
    - All nodes default to 'Kinase'
    - Nodes that are only targets of 'regulates' are labeled 'mRNA'

    Parameters:
        edge_df (pd.DataFrame): Must have columns ['Source', 'Target', 'Interaction']

    Returns:
        pd.DataFrame: DataFrame with columns ['Node', 'Type']
    """
    node_roles = {}

    for _, row in edge_df.iterrows():
        src, tgt, interaction = row["Source"], row["Target"], row["Interaction"]

        if interaction == "regulates":
            node_roles[src] = "TF"
            node_roles[tgt] = "Kinase" if src not in node_roles else node_roles[tgt]
        else:
            node_roles[src] = "Kinase"
            node_roles[tgt] = "Kinase"

    return pd.DataFrame([
        {"Node": node, "Type": node_type}
        for node, node_type in node_roles.items()
    ])


if __name__ == "__main__":

    # Path to the Excel file of mRNA-TF optimization results
    file_path = ROOT / "data" / "tfopt_results.xlsx"

    # Call the function to map optimization results
    mapped_df = map_optimization_results(file_path)

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
    os.rename('nodes.csv', ROOT / "data" / 'nodes.csv')
    os.rename('mapped_TF_mRNA_phospho.csv', ROOT / "data" / 'mapping.csv')
    os.rename('mapping_table.csv', ROOT / "data" / 'mapping_.csv')

    logger.info(f"Mapping files for merging with ODE results & further use in Cytoscape")
    for fpath in [ROOT / "data" / 'nodes.csv',
                  ROOT / "data" / 'mapping.csv',
                  ROOT / "data" / 'mapping_.csv']:
        logger.info(f"{fpath.as_uri()}")

# Note: The following comment is an example of how to handle missing data
# PAK2 is not included in the mapping file because it doesn't exist in the input4.csv file.

""" 
PAK2 doesn't exist in CollectTRI (so also not in input4.csv) but it is  
in the GeneID column of the phospho interaction file. 
No mRNA-TF estimation is possible for PAK2.  
Hence, it is not included in the output mapped file for ODE modelling.
"""
