import shutil

from config.helpers import location
from tfopt.local.config.constants import parse_args, OUT_DIR, OUT_FILE, ODE_DATA_DIR
from tfopt.local.config.logconf import setup_logger
from tfopt.local.utils.iodata import organize_output_files, create_report, summarize_stats
from tfopt.local.exporter.plotout import plot_estimated_vs_observed
from tfopt.local.exporter.sheetutils import save_results_to_excel
from tfopt.local.objfn.minfn import compute_predictions
from tfopt.local.opt.optrun import run_optimizer
from tfopt.local.optcon.filter import load_and_filter_data, prepare_data
from tfopt.local.utils.params import get_optimization_parameters, postprocess_results
from tfopt.fitanalysis.helper import Plotter

logger = setup_logger()

def main():
    """
    Main function to run the mRNA-TF optimization problem.
    This function orchestrates the loading of data, preparation of fixed arrays,
    setting up optimization parameters, running the optimization, and post-processing
    the results. It also handles the visualization and saving of results.
    The function starts by parsing command line arguments, loading and filtering
    data, preparing the data and fixed arrays, setting up optimization parameters,
    running the optimization, and finally post-processing the results.
    """
    logger.info("[Local Optimization] mRNA-TF Optimization Problem Started")

    # STEP 0: Parse command line arguments.
    lb, ub, loss_type = parse_args()

    # STEP 1: Load and filter the data.
    gene_ids, expr_matrix, expr_time_cols, tf_ids, tf_protein, tf_psite_data, tf_psite_labels, tf_time_cols, reg_map = \
        load_and_filter_data()

    # logger.info(f"Names of mRNAs: {gene_ids}")
    # logger.info(f"Names of TFs: {tf_ids}")
    # logger.info(f"Names of TFProtein: {tf_protein}")
    # logger.info(f"Names of TFPsiteData: {tf_psite_data}")
    # logger.info(f"Names of TFPsiteLabels: {tf_psite_labels}")
    # logger.info(f"Names of TFTimeCols: {tf_time_cols}")
    # logger.info(f"Names of RegMap: {reg_map}")

    # summarize_stats()

    # STEP 2: Prepare data and build fixed arrays.
    fixed_arrays, T_use = prepare_data(gene_ids, expr_matrix, tf_ids, tf_protein, tf_psite_data,
                                        tf_psite_labels, tf_time_cols, reg_map)
    expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, n_psite_max, psite_labels_arr, num_psites = fixed_arrays

    # n_genes = expression_matrix.shape[0]
    # n_TF = tf_protein_matrix.shape[0]

    # logger.info(f"Number of messenger RNAs: {n_genes}")
    # logger.info(f"Number of Transcription Factors: {n_TF}")

    # STEP 3: Set up optimization parameters.
    x0, n_alpha, beta_start_indices, bounds, no_psite_tf, n_genes, n_TF, num_psites, lin_cons, T_use = \
        get_optimization_parameters(expression_matrix, tf_protein_matrix, n_reg, T_use,
                                psite_labels_arr, num_psites, lb, ub)

    # STEP 4: Run the optimization.
    result = run_optimizer(x0, bounds, lin_cons, expression_matrix, regulators, tf_protein_matrix, psite_tensor,
                           n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type)

    logger.info("--- Best Solution ---")
    logger.info(f"Objective Value (F): {result.fun}")

    # STEP 5: Post-process results and output.
    final_x, final_alpha, final_beta = postprocess_results(result, n_alpha, n_genes, n_reg, beta_start_indices,
                                                           num_psites, reg_map, gene_ids, tf_ids, psite_labels_arr)

    # Compute predictions and plot results.
    predictions = compute_predictions(final_x, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                                      beta_start_indices, num_psites)
    plot_estimated_vs_observed(predictions, expression_matrix, gene_ids, expr_time_cols, regulators,
                               tf_protein_matrix, tf_ids, num_targets=n_genes)

    # Save results to Excel.
    save_results_to_excel(gene_ids, tf_ids, final_alpha, final_beta, psite_labels_arr, expression_matrix,
                          predictions, result.fun, reg_map)

    # Generate plots.
    plotter = Plotter(OUT_FILE, OUT_DIR)
    plotter.plot_alpha_distribution()
    plotter.plot_beta_barplots()
    plotter.plot_heatmap_abs_residuals()
    plotter.plot_goodness_of_fit()
    plotter.plot_kld()
    plotter.plot_pca()
    plotter.plot_boxplot_alpha()
    plotter.plot_boxplot_beta()
    plotter.plot_cdf_alpha()
    plotter.plot_cdf_beta()
    plotter.plot_time_wise_residuals()

    # Copy output file to the ODE_DATA_DIR.
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)

    # Organize output files and create a report.
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)

    logger.info(f'[Local] Report & Results {location(str(OUT_DIR))}')

    # Click to open the report in a web browser.
    for fpath in [OUT_DIR / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

if __name__ == "__main__":
    main()