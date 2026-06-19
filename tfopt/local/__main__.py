from config.helpers import location
from tfopt.local.config.constants import parse_args, OUT_FILE, INPUT1, INPUT3, INPUT4, _CFG
from tfopt.local.config.logconf import setup_logger
from tfopt.local.utils.iodata import organize_output_files, create_report
from tfopt.local.exporter.plotout import plot_estimated_vs_observed, plot_multistart_summary_runtime_overlay
from tfopt.local.exporter.sheetutils import save_results_to_excel, export_multistart_results, \
    save_multistart_solutions_npz
from tfopt.local.objfn.minfn import compute_predictions
from tfopt.local.opt.optrun import run_optimizer, run_optimizer_multistart, MultiStartConfig, _get_constraint_violation
from tfopt.local.optcon.filter import load_and_filter_data, prepare_data
from tfopt.local.utils.params import get_optimization_parameters, postprocess_results
from tfopt.fitanalysis.helper import Plotter
from common.utils import latexit
from common.results import (
    attach_file_console_logger, ensure_result_dir, populate_standard_subdirs,
    write_command, write_metadata, write_resolved_config,
)

logger = setup_logger()


def main():
    """
    Main function to run the mRNA-TF optimization problem using gradient based optimizer.
    """
    logger.info("[Local Optimization] mRNA-TF Optimization Problem Started")

    # STEP 0: Parse command line arguments.
    lb, ub, loss_type, out_dir = parse_args()
    result_dirs = ensure_result_dir(out_dir)
    out_dir = result_dirs["root"]
    out_file = out_dir / OUT_FILE.name
    attach_file_console_logger(logger, out_dir)
    write_command(out_dir)
    write_resolved_config(out_dir, _CFG)
    write_metadata(
        out_dir,
        workflow="tfopt.local",
        args={"lower_bound": lb, "upper_bound": ub, "loss_type": loss_type},
        inputs=[INPUT1, INPUT3, INPUT4],
    )

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
    cfg = MultiStartConfig(
        n_starts=48,  # start with 32–64
        n_jobs=-1,  # all cores
        seed=123,
        jitter_frac=0.05,
        p_random=0.35,
    )

    result, all_results = run_optimizer_multistart(
        x0, bounds, lin_cons,
        expression_matrix, regulators, tf_protein_matrix, psite_tensor,
        n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type,
        run_optimizer_func=run_optimizer,
        cfg=cfg,
        polish=True,
    )

    logger.info(f"[Multistart] best fun={result.fun} success={getattr(result, 'success', None)} "
                f"cv={_get_constraint_violation(result)} start_id={getattr(result, 'start_id', None)}")

    # Save multistart results to CSV.
    export_multistart_results(all_results).to_csv(out_dir / "multistart_summary.csv", index=False)

    # Save multistart solutions to NPZ.
    save_multistart_solutions_npz(
        all_results,
        out_dir / "multistart_params.npz",
    )

    # Save waterfall plot.
    plot_multistart_summary_runtime_overlay(
        out_dir / "multistart_summary.csv",
        out_path=out_dir / "multistart_fun_vs_rank_runtime.png",
        figsize=(8, 8),
    )

    logger.info("--- Best Solution ---")
    logger.info(f"Objective Value (F): {result.fun}")

    # STEP 5: Post-process results and output.
    final_x, final_alpha, final_beta = postprocess_results(result, n_alpha, n_genes, n_reg, beta_start_indices,
                                                           num_psites, reg_map, gene_ids, tf_ids, psite_labels_arr)

    # Compute predictions and plot results.
    predictions = compute_predictions(final_x, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes,
                                      beta_start_indices, num_psites)
    plot_estimated_vs_observed(predictions, expression_matrix, gene_ids, expr_time_cols, regulators,
                               tf_protein_matrix, tf_ids, num_targets=n_genes, save_path=out_dir)

    # Save results to Excel.
    save_results_to_excel(gene_ids, tf_ids, final_alpha, final_beta, psite_labels_arr, expression_matrix,
                          predictions, result.fun, reg_map, filename=out_file)

    # Generate plots.
    plotter = Plotter(out_file, out_dir)
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

    # LateX the results
    latexit.main(out_dir)

    # Organize output files and create a report.
    organize_output_files(out_dir)
    create_report(out_dir)

    logger.info(f'[Local] Report & Results {location(str(out_dir))}')

    # Click to open the report in a web browser.
    for fpath in [out_dir / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

    populate_standard_subdirs(out_dir)


if __name__ == "__main__":
    main()
