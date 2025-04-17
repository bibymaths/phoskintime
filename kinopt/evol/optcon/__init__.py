from kinopt.evol.optcon.construct import pipeline
from kinopt.evol.config import scaling_method, split_point, segment_points, estimate_missing_kinases, kinase_to_psites, time_series_columns
from kinopt.evol.config.constants import INPUT1, INPUT2

# The pipeline function is responsible for constructing the necessary data structures and performing
# the computations required for the optimization process.
# The pipeline function takes several parameters, including input files, scaling method,
# split point, segment points, and options for estimating missing kinases and kinase-to-psite mappings.
# The pipeline function returns several outputs, including:
# - full_hgnc_df: DataFrame containing the full HGNC data.
# - interaction_df: DataFrame containing interaction data.
# - observed: Observed time-series data.
# - P_initial: Initial mapping of gene-psite pairs to kinase relationships and time-series data.
# - P_initial_array: Array containing observed time-series data for gene-psite pairs.
# - K_index: Mapping of kinases to their respective psite data.
# - K_array: Array containing time-series data for kinase-psite combinations.
# - beta_counts: Mapping of kinase indices to the number of associated psites.
# - gene_psite_counts: Number of kinases per gene-psite combination.
# - n: Number of decision variables in the optimization problem.

(full_hgnc_df, interaction_df, observed, P_initial, P_initial_array,
  K_index, K_array, beta_counts, gene_psite_counts, n) =  (
    pipeline(INPUT1, INPUT2, time_series_columns, scaling_method, split_point,
           segment_points, estimate_missing_kinases, kinase_to_psites))
