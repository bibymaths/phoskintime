import numpy as np
import pandas as pd
from abopt.nsgaii.utils.iodata import apply_scaling

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
    Loads two CSV files, applies scaling to the time-series columns of `input1`, and subsets/merges them
    depending on `estimate_missing_kinases`.

    Returns:
        full_hgnc_df (pd.DataFrame): The scaled data from input1
        interaction_df (pd.DataFrame): The subset/merged DataFrame from input2
        observed (pd.DataFrame): Subset of full_hgnc_df merged with interaction_df
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
        ).drop(columns=["max", "min"])

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
        ).drop(columns=["max", "min"])

    return full_hgnc_df, interaction_df, observed


def _build_p_initial(
    interaction_df: pd.DataFrame,
    full_hgnc_df: pd.DataFrame,
    time_series_cols: list[str]
):
    """
    Creates P_initial_array and a dictionary P_initial with
    gene-psite → { 'Kinases': [...], 'TimeSeries': [...] }
    """
    P_initial = {}
    P_initial_array = []
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
        if not observed_data.empty:
            time_series = observed_data[time_series_cols].values.flatten()
        else:
            time_series = np.ones(num_time_points)

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
    Creates K_array, K_index, and beta_counts for the kinases described
    in interaction_df. If 'estimate_missing_kinases' is True, adds synthetic
    psites for any missing kinase in 'kinase_to_psites'.
    """
    K_index = {}
    K_array = []
    beta_counts = {}

    synthetic_counter = 1

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

                if kinase not in K_index:
                    K_index[kinase] = []
                K_index[kinase].append((psite, time_series))

                beta_counts[idx] = 1
        elif estimate_missing_kinases:

            if kinase not in K_index:
                K_index[kinase] = []

            add_psites = kinase_to_psites.get(kinase, 1)
            for _ in range(add_psites):
                synthetic_label = f"P{synthetic_counter}"
                synthetic_counter += 1

                synthetic_time_series = np.ones(len(time), dtype=np.float64)  # or some default
                K_array.append(synthetic_time_series)
                K_index[kinase].append((synthetic_label, synthetic_time_series))
                beta_counts[len(K_array) - 1] = 1

    K_array = np.array(K_array)

    return K_array, K_index, beta_counts


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
    High-level function that calls all the smaller steps:
      1) load & scale data
      2) subset & merge
      3) build P_initial
      4) build K_array
    Returns the data structures needed for the next steps.
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
        full_hgnc_df,         # pd.DataFrame
        interaction_df,       # pd.DataFrame
        observed,             # pd.DataFrame
        P_initial,            # dict
        P_initial_array,      # np.ndarray
        K_array,              # np.ndarray
        K_index,              # dict
        beta_counts,          # dict
        gene_psite_counts,    # list
        n                     # int
    )

