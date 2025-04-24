import shutil
from kinopt.local.config.constants import parse_args, OUT_DIR, OUT_FILE, ODE_DATA_DIR
from kinopt.local.config.helpers import location
from kinopt.local.exporter.sheetutils import output_results
from kinopt.local.opt.optrun import run_optimization
from kinopt.local.optcon.construct import check_kinases
from kinopt.local.utils.iodata import load_and_scale_data, organize_output_files, create_report
from kinopt.local.objfn import objective_wrapper
from kinopt.local.optcon import (build_K_data, build_constraints, build_P_initial, init_parameters,
                                 compute_time_weights, precompute_mappings, convert_to_sparse)
from kinopt.local.utils.params import compute_metrics, extract_parameters
from kinopt.local.config.logconf import setup_logger
from kinopt.optimality.KKT import post_optimization_results
from kinopt.fitanalysis import optimization_performance
logger = setup_logger()


def main():
    """
    Main function to run the local optimization for kinase phosphorylation time series data.

    It performs the following steps:
    1. Sets up logging.
    2. Parses command-line arguments.
    3. Loads and scales the data.
    4. Builds the initial protein group data matrix.
    5. Builds the kinase data matrix.
    6. Precomputes mappings for optimization.
    7. Initializes parameters for optimization.
    8. Computes time weights for the optimization.
    9. Builds constraints for the optimization.
    10. Defines the objective function wrapper.
    11. Runs the optimization using the specified method.
    12. Extracts optimized parameters from the optimization results.
    13. Computes metrics for the optimized parameters.
    14. Outputs results to an Excel file.
    15. Copies the output file to a specified directory.
    16. Analyzes optimization performance.
    17. Organizes output files and creates a report.
    18. Logs the completion of the process.

    The function takes no parameters and returns nothing.
    """
    # Set up logging.
    logger.info('[Local Optimization] Started - Kinase Phosphorylation Time Series')

    # Check for the missing kinases in the input files.
    # From the input2.csv file, it checks if the kinases are present in the input1.csv file.
    check_kinases()

    # Parse arguments.
    lb, ub, loss_type, estimate_missing, scaling_method, split_point, seg_points, opt_method = parse_args()

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

    # Define objective wrapper.
    obj_fun = lambda p: objective_wrapper(p, P_init_dense, t_max, gene_alpha_starts, gene_kinase_counts,
                                          gene_kinase_idx, total_alpha, kinase_beta_starts, kinase_beta_counts,
                                          K_data, K_indices, K_indptr, time_weights, loss_type)

    # Run optimization.
    result, optimized_params = run_optimization(obj_fun, params_initial, opt_method, bounds, constraints)

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
                   result, mse, rmse, mae, mape, r_squared)

    # Copy output file to ODE data directory.
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)

    # Analyze optimization performance.
    post_optimization_results()
    optimization_performance()

    # Organize output files and create a report.
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)

    logger.info(f'Report & Results {location(str(OUT_DIR))}')

    # Click to open the report in a web browser.
    for fpath in [OUT_DIR / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

if __name__ == "__main__":
    main()