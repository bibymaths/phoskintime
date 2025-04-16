from kinopt.evol.optcon.construct import pipeline
from kinopt.evol.config import scaling_method, split_point, segment_points, estimate_missing_kinases, kinase_to_psites, time_series_columns
from kinopt.evol.config.constants import INPUT1, INPUT2

(full_hgnc_df, interaction_df, observed, P_initial, P_initial_array,
  K_index, K_array, beta_counts, gene_psite_counts, n) =  (
    pipeline(INPUT1, INPUT2, time_series_columns, scaling_method, split_point,
           segment_points, estimate_missing_kinases, kinase_to_psites))
