import csv
from collections import defaultdict

import numpy as np
import pandas as pd

from kinopt.evol.config.constants import INPUT2, INPUT1
from kinopt.evol.utils.iodata import apply_scaling

from kinopt.evol.config.logconf import setup_logger

logger = setup_logger()


def _load_and_scale_data(
        input1_path: str,
        input2_path: str,
        time_series_columns: list[str],
        scaling_method: str,
        split_point: float,
        segment_points: list[float],
        estimate_missing_kinases: bool
):
    """
    Function to load and scale data from two CSV files.

    Args:
        input1_path (str): Path to the first CSV file (HGNC data).
        input2_path (str): Path to the second CSV file (kinase interactions).
        time_series_columns (list[str]): List of time series columns to extract.
        scaling_method (str): Method for scaling the data.
        split_point (float): Split point for scaling.
        segment_points (list[float]): Segment points for scaling.
        estimate_missing_kinases (bool): Flag to estimate missing kinases.
    Returns:
        full_hgnc_df (pd.DataFrame): The scaled data from input1.
        interaction_df (pd.DataFrame): The subset/merged DataFrame from input2.
        observed (pd.DataFrame): Subset of full_hgnc_df merged with interaction_df.
    """
    # 1) Load the data
    full_hgnc_df = pd.read_csv(input1_path)
    interaction_df = pd.read_csv(input2_path, header=0)

    # 2) Apply scaling (this uses your existing apply_scaling function)
    #    You’ll have to define or import apply_scaling from where it’s defined
    full_hgnc_df = apply_scaling(
        full_hgnc_df,
        time_series_columns,
        scaling_method,
        split_point,
        segment_points
    )

    # 3) Subset data
    if estimate_missing_kinases:
        observed = full_hgnc_df.merge(
            interaction_df.iloc[:, :2],
            on=["GeneID", "Psite"]
        )

        # Convert string of kinases like "{K1,K2}" → ["K1","K2"]
        interaction_df['Kinase'] = (
            interaction_df['Kinase'].str.strip('{}')
            .apply(lambda x: [k.strip() for k in x.split(',')])
        )
    else:
        # Filter out kinases not present in the full_hgnc_df
        interaction_df = interaction_df[
            interaction_df['Kinase'].apply(
                lambda k: all(
                    kinase in set(full_hgnc_df['GeneID'][1:])
                    for kinase in k.strip('{}').split(',')
                )
            )
        ]
        # Convert kinases to list
        interaction_df['Kinase'] = (
            interaction_df['Kinase'].str.strip('{}')
            .apply(lambda x: [k.strip() for k in x.split(',')])
        )
        observed = full_hgnc_df.merge(
            interaction_df.iloc[:, :2],
            on=["GeneID", "Psite"]
        )

    return full_hgnc_df, interaction_df, observed


def _build_p_initial(
        interaction_df: pd.DataFrame,
        full_hgnc_df: pd.DataFrame,
        time_series_cols: list[str]
):
    """
    Function to build the P_initial structure.

    Args:
        interaction_df (pd.DataFrame): DataFrame containing kinase interactions.
        full_hgnc_df (pd.DataFrame): DataFrame containing scaled HGNC data.
        time_series_cols (list[str]): List of time series columns to extract.

    Returns:
        P_initial (dict): Dictionary mapping gene-psite pairs to kinase relationships and time-series data.
        P_initial_array (np.ndarray): Array containing observed time-series data for gene-psite pairs.
    """
    P_initial = {}
    P_initial_array = []
    time = [f'x{i}' for i in range(1, 15)]
    num_time_points = len(time_series_cols)

    for _, row in interaction_df.iterrows():
        gene = row['GeneID']
        psite = row['Psite']
        kinases = [k.strip() for k in row['Kinase']]  # ensure no whitespace

        # Retrieve time series data for gene-psite
        observed_data = full_hgnc_df[
            (full_hgnc_df['GeneID'] == gene) &
            (full_hgnc_df['Psite'] == psite)
            ]
        # Grab the time series data for the gene and phosphorylation site
        # Check in observed_data if that (gene, psite) combination exists and get time series
        match = observed_data[(observed_data['GeneID'] == gene) & (observed_data['Psite'] == psite)]
        if not match.empty:
            time_series = match.iloc[0][time].values.astype(np.float64)

        P_initial_array.append(time_series)

        P_initial[(gene, psite)] = {
            'Kinases': kinases,
            'TimeSeries': time_series
        }

    # Convert to numpy array
    P_initial_array = np.array(P_initial_array)
    return P_initial, P_initial_array


def _build_k_array(
        interaction_df: pd.DataFrame,
        full_hgnc_df: pd.DataFrame,
        time: list[str],
        estimate_missing_kinases: bool,
        kinase_to_psites: dict[str, int]
):
    """
    Function to build the Kinase time series structure.

    Args:
        interaction_df (pd.DataFrame): DataFrame containing kinase interactions.
        full_hgnc_df (pd.DataFrame): DataFrame containing scaled HGNC data.
        time (list[str]): List of time series columns to extract.
        estimate_missing_kinases (bool): Flag to estimate missing kinases.
        kinase_to_psites (dict[str, int]): Dictionary mapping kinases to their respective psites.

    Returns:
        K_index (dict): Mapping of kinases to their respective psite data.
        K_array (np.ndarray): Array containing time-series data for kinase-psite combinations.
        beta_counts (dict): Mapping of kinase indices to the number of associated psites.

    """
    K_index = {}
    K_array = []
    beta_counts = {}

    synthetic_counter = 1
    synthetic_rows = []

    # Unique kinases from the DataFrame's 'Kinase' column
    unique_kinases = interaction_df['Kinase'].explode().unique()

    for kinase in unique_kinases:
        # Subset rows in full_hgnc_df for that kinase
        kinase_psite_data = full_hgnc_df[
            full_hgnc_df['GeneID'] == kinase
            ][['Psite'] + time]

        if not kinase_psite_data.empty:
            # Iterate over all psites for this kinase
            for _, row in kinase_psite_data.iterrows():
                psite = row['Psite']
                time_series = np.array(row[time].values, dtype=np.float64)
                idx = len(K_array)
                K_array.append(time_series)
                K_index.setdefault(kinase, []).append((psite, time_series))
                beta_counts[idx] = 1

        elif estimate_missing_kinases:
            synthetic_label = f"P{synthetic_counter}"
            synthetic_counter += 1
            # Get protein time series for this kinase where 'Psite' is empty or NaN
            protein_level_df = full_hgnc_df[(full_hgnc_df['GeneID'] == kinase) & (full_hgnc_df['Psite'].isna())]
            if not protein_level_df.empty:
                # Adding the non-psite time series
                synthetic_ts = np.array(protein_level_df.iloc[0][time].values, dtype=np.float64)
            idx = len(K_array)
            K_array.append(synthetic_ts)
            K_index.setdefault(kinase, []).append((synthetic_label, synthetic_ts))
            beta_counts[idx] = 1
            synthetic_rows.append(idx)

    # Finalize K_array
    K_array = np.array(K_array)

    return K_index, K_array, beta_counts


def pipeline(
        input1_path: str,
        input2_path: str,
        time_series_columns: list[str],
        scaling_method: str,
        split_point: float,
        segment_points: list[float],
        estimate_missing_kinases: bool,
        kinase_to_psites: dict[str, int]
):
    """
    Function to run the entire pipeline for loading and processing data.

    Args:
        input1_path (str): Path to the first CSV file (HGNC data).
        input2_path (str): Path to the second CSV file (kinase interactions).
        time_series_columns (list[str]): List of time series columns to extract.
        scaling_method (str): Method for scaling the data.
        split_point (float): Split point for scaling.
        segment_points (list[float]): Segment points for scaling.
        estimate_missing_kinases (bool): Flag to estimate missing kinases.
        kinase_to_psites (dict[str, int]): Dictionary mapping kinases to their respective psites.

    Returns:
        full_hgnc_df (pd.DataFrame): The scaled data from input1.
        interaction_df (pd.DataFrame): The subset/merged DataFrame from input2.
        observed (pd.DataFrame): Subset of full_hgnc_df merged with interaction_df.
        P_initial (dict): Dictionary mapping gene-psite pairs to kinase relationships and time-series data.
        P_initial_array (np.ndarray): Array containing observed time-series data for gene-psite pairs.
        K_array (np.ndarray): Array containing time-series data for kinase-psite combinations.
        K_index (dict): Mapping of kinases to their respective psite data.
        beta_counts (dict): Mapping of kinase indices to the number of associated psites.
        gene_psite_counts (list): List of counts of psites for each gene.
        n (int): Number of unique gene-psite pairs.
    """
    # 1) Load and scale
    full_hgnc_df, interaction_df, observed = _load_and_scale_data(
        input1_path=input1_path,
        input2_path=input2_path,
        time_series_columns=time_series_columns,
        scaling_method=scaling_method,
        split_point=split_point,
        segment_points=segment_points,
        estimate_missing_kinases=estimate_missing_kinases
    )

    # 2) Build P_initial
    P_initial, P_initial_array = _build_p_initial(
        interaction_df, full_hgnc_df, time_series_columns
    )
    n = P_initial_array.size

    # 3) Build K_array
    K_array, K_index, beta_counts = _build_k_array(
        interaction_df=interaction_df,
        full_hgnc_df=full_hgnc_df,
        time=time_series_columns,
        estimate_missing_kinases=estimate_missing_kinases,
        kinase_to_psites=kinase_to_psites
    )

    # 4) gene_psite_counts for alpha parameters
    gene_psite_counts = [len(data['Kinases']) for data in P_initial.values()]

    return (
        full_hgnc_df,
        interaction_df,
        observed,
        P_initial,
        P_initial_array,
        K_array,
        K_index,
        beta_counts,
        gene_psite_counts,
        n
    )


def load_geneid_to_psites(input1_path=INPUT1):
    """
    Function to load geneid to psite mapping from input1.csv.
    Args:
        input1_path (str): Path to the first CSV file (HGNC data).
    Returns:
        geneid_psite_map (dict): Dictionary mapping gene IDs to sets of psites.
    """
    geneid_psite_map = defaultdict(set)
    with open(input1_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            geneid = row['GeneID'].strip()
            psite = row['Psite'].strip()
            if geneid and psite:
                geneid_psite_map[geneid].add(psite)
    return geneid_psite_map


def get_unique_kinases(input2_path=INPUT2):
    """
    Function to extract unique kinases from input2.csv.
    Args:
        input2_path (str): Path to the second CSV file (kinase interactions).
    Returns:
        kinases (set): Set of unique kinases extracted from the input2 file.
    """
    kinases = set()
    with open(input2_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kinase_raw = row['Kinase'].strip()
            # Remove surrounding curly braces if present
            if kinase_raw.startswith("{") and kinase_raw.endswith("}"):
                kinase_raw = kinase_raw[1:-1]
            # Split if it's a list like {KIN1,KIN2}
            for k in kinase_raw.split(","):
                k = k.strip()
                if k:
                    kinases.add(k)
    return kinases


def check_kinases():
    """
    Function to check if kinases from input2.csv are present in input1.csv.

    Args:
        None

    Returns:
        None
    """
    geneid_to_psites = load_geneid_to_psites()
    kinases = get_unique_kinases()

    logger.info("--- Kinase Check in input1.csv ---")
    for kinase in sorted(kinases):
        if kinase in geneid_to_psites:
            psites = sorted(geneid_to_psites[kinase])
            logger.info(f"{kinase}: Found, Psites in input1 -> {psites}")
        else:
            logger.info(f"{kinase}: Missing")

    logger.info("--- Summary ---")
    found = sum(1 for k in kinases if k in geneid_to_psites)
    missing = len(kinases) - found
    logger.info(f"Total unique kinases: {len(kinases)}")
    logger.info(f"Found: {found}")
    logger.info(f"Missing: {missing}")
