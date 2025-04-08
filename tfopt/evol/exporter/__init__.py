from tfopt.evol.exporter.plotout import compute_predictions, plot_estimated_vs_observed
from tfopt.evol.exporter.sheetutils import save_results_to_excel

# -------------------------------
# Post-Processing: Predictions, Plotting, and Saving Results
# -------------------------------
def post_processing(final_x, regulators, protein_mat, psite_tensor, n_reg, n_TF, T_use, n_mRNA,
                    beta_start_indices, num_psites, mRNA_ids, mRNA_mat, mRNA_time_cols, TF_ids,
                    final_alpha, final_beta, psite_labels_arr, best_objectives, reg_map):
    predictions = compute_predictions(final_x, regulators, protein_mat, psite_tensor, n_reg,
                                      T_use, n_mRNA, beta_start_indices, num_psites)
    plot_estimated_vs_observed(predictions, mRNA_mat, mRNA_ids, mRNA_time_cols, regulators,
                               protein_mat, TF_ids, n_TF)
    save_results_to_excel(mRNA_ids, TF_ids, final_alpha, final_beta, psite_labels_arr,
                          mRNA_mat, predictions, best_objectives, reg_map)
