from functools import partial

from kinopt.local.config.constants import parse_args, OUT_FILE, INPUT1, INPUT2, _CFG
from kinopt.local.config.helpers import location
from kinopt.local.exporter.plotout import export_outcomes_to_csv, plot_multistart_summary_runtime_overlay
from kinopt.local.exporter.sheetutils import output_results, export_params_npz
from kinopt.local.opt.optrun import multistart_run_optimization
from kinopt.local.optcon.construct import check_kinases
from kinopt.local.utils.iodata import load_and_scale_data, organize_output_files, create_report
from kinopt.local.objfn import objective_wrapper
from kinopt.local.optcon import (build_K_data, build_constraints, build_P_initial, init_parameters,
                                 compute_time_weights, precompute_mappings, convert_to_sparse)
from kinopt.local.utils.params import compute_metrics, extract_parameters
from kinopt.local.config.logconf import setup_logger
from kinopt.optimality.KKT import post_optimization_results
from kinopt.fitanalysis import optimization_performance
from common.utils import latexit
from common.results import (
    attach_file_console_logger, ensure_result_dir, populate_standard_subdirs,
    write_command, write_metadata, write_resolved_config,
)

logger = setup_logger()


def main():
    """
    Main function to run the local optimization for kinase phosphorylation time series data.
    """

    # Set up logging.
    logger.info('[Local Optimization] Started - Kinase Phosphorylation Time Series')

    # Check for the missing kinases in the input files.
    # From the input2.csv file, it checks if the kinases are present in the input1.csv file.
    check_kinases()

    # Parse arguments.
    lb, ub, loss_type, estimate_missing, scaling_method, split_point, seg_points, opt_method, out_dir = parse_args()
    result_dirs = ensure_result_dir(out_dir)
    out_dir = result_dirs["root"]
    out_file = out_dir / OUT_FILE.name
    attach_file_console_logger(logger, out_dir)
    write_command(out_dir)
    write_resolved_config(out_dir, _CFG)
    write_metadata(
        out_dir,
        workflow="kinopt.local",
        args={
            "lower_bound": lb, "upper_bound": ub, "loss_type": loss_type,
            "estimate_missing_kinases": estimate_missing, "scaling_method": scaling_method,
            "split_point": split_point, "segment_points": seg_points, "method": opt_method,
        },
        inputs=[INPUT1, INPUT2],
    )

    # Load and scale data.
    full_df, interact_df, _ = load_and_scale_data(estimate_missing, scaling_method, split_point, seg_points)

    # Build protein group data matrix.
    P_initial, P_array = build_P_initial(full_df, interact_df)

    # Build kinase data matrix.
    K_index, K_array, beta_counts = build_K_data(full_df, interact_df, estimate_missing)

    # Convert kinase matrix to sparse format.
    K_sparse, K_data, K_indices, K_indptr = convert_to_sparse(K_array)

    # Precompute mappings for optimization.
    (unique_kinases, gene_kinase_counts, gene_alpha_starts, gene_kinase_idx, total_alpha,
     kinase_beta_counts, kinase_beta_starts) = precompute_mappings(P_initial, K_index)

    # Initialize parameters initial values.
    params_initial, bounds = init_parameters(total_alpha, lb, ub, kinase_beta_counts)

    # Compute time weights.
    t_max, P_init_dense, time_weights = compute_time_weights(P_array, loss_type)

    # Build constraints.
    constraints = build_constraints(opt_method, gene_kinase_counts, unique_kinases, total_alpha, kinase_beta_counts,
                                    len(params_initial))

    # Deprecated single start
    # obj_fun = lambda p: objective_wrapper(p, P_init_dense, t_max, gene_alpha_starts, gene_kinase_counts,
    #                                       gene_kinase_idx, total_alpha, kinase_beta_starts, kinase_beta_counts,
    #                                       K_data, K_indices, K_indptr, time_weights, loss_type)

    # Multistart optimization
    obj_fun = partial(
        objective_wrapper,
        P_init_dense=P_init_dense,
        t_max=t_max,
        gene_alpha_starts=gene_alpha_starts,
        gene_kinase_counts=gene_kinase_counts,
        gene_kinase_idx=gene_kinase_idx,
        total_alpha=total_alpha,
        kinase_beta_starts=kinase_beta_starts,
        kinase_beta_counts=kinase_beta_counts,
        K_data=K_data,
        K_indices=K_indices,
        K_indptr=K_indptr,
        time_weights=time_weights,
        loss_type=loss_type
    )


    # Deprecated single start
    # result, optimized_params = run_optimization(obj_fun, params_initial, opt_method, bounds, constraints)

    # Multistart optimization
    result, optimized_params, outcomes = multistart_run_optimization(
        obj_fun=obj_fun,
        params_initial=params_initial,
        opt_method=opt_method,
        bounds=bounds,
        constraints=constraints,
        n_starts=64,  # adjust; 16–64 typical
        n_jobs=-1,  # all cores
        base_seed=20260115,  # reproducible
        init_strategy="hybrid",
        jitter_scale=0.10,  # conservative perturbation
        prefer_feasible=True,
        logger=logger
    )

    # Save outcomes
    export_outcomes_to_csv(
        outcomes,
        out_dir / "multistart_summary.csv"
    )

    # Save optimized parameters for each start
    export_params_npz(outcomes, out_dir / "multistart_params.npz")

    # Save runtime vs objective function value plot
    plot_multistart_summary_runtime_overlay(
        out_dir / "multistart_summary.csv",
        out_path=out_dir / "multistart_fun_vs_rank_runtime.png",
        figsize=(8, 8),
    )

    # Extract optimized parameters.
    alpha_values, beta_values = extract_parameters(P_initial, gene_kinase_counts, total_alpha, unique_kinases, K_index,
                                                   optimized_params)
    # Compute metrics.
    P_estimated, residuals, mse, rmse, mae, mape, r_squared = compute_metrics(optimized_params, P_init_dense, t_max,
                                                                              gene_alpha_starts, gene_kinase_counts,
                                                                              gene_kinase_idx,
                                                                              total_alpha, kinase_beta_starts,
                                                                              kinase_beta_counts,
                                                                              K_data, K_indices, K_indptr)

    # Output results.
    output_results(P_initial, P_init_dense, P_estimated, residuals, alpha_values, beta_values,
                   result, mse, rmse, mae, mape, r_squared, filename=out_file, out_dir=out_dir)


    # Analyze optimization performance using the selected result directory.
    import kinopt.optimality.KKT as kkt_module
    import kinopt.fitanalysis.__main__ as fitanalysis_module
    import kinopt.fitanalysis.helpers.postfit as postfit_module
    kkt_module.OUT_FILE = out_file
    kkt_module.OUT_DIR = out_dir
    fitanalysis_module.OUT_FILE = out_file
    fitanalysis_module.OUT_DIR = out_dir
    postfit_module.OUT_DIR = out_dir
    post_optimization_results()
    optimization_performance()

    # LateX the results.
    latexit.main(out_dir)

    # Organize output files and create a report.
    organize_output_files(out_dir)
    create_report(out_dir)

    logger.info(f'Report & Results {location(str(out_dir))}')

    # Click to open the report in a web browser.
    for fpath in [out_dir / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

    populate_standard_subdirs(out_dir)


if __name__ == "__main__":
    main()
