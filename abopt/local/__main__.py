import shutil
from abopt.local.config.constants import parse_args, OUT_DIR, OUT_FILE, ODE_DATA_DIR
from abopt.local.config.helpers import location
from abopt.local.exporter.sheetutils import output_results
from abopt.local.opt.optrun import run_optimization
from abopt.local.utils.iodata import load_and_scale_data, organize_output_files, create_report
from abopt.local.objfn import objective_wrapper
from abopt.local.optcon import (build_K_data, build_constraints, build_P_initial, init_parameters,
                                compute_time_weights, precompute_mappings, convert_to_sparse)
from abopt.local.utils.params import compute_metrics, extract_parameters
from abopt.local.config.logconf import setup_logger
from abopt.optimality.KKT import post_optimization_results
from abopt.fitanalysis import optimization_performance
logger = setup_logger()


def main():
    logger.info('[Local Optimization] Started')
    # Parse arguments.
    lb, ub, loss_type, estimate_missing, scaling_method, split_point, seg_points, opt_method = parse_args()
    # Load and scale data.
    full_df, interact_df, _ = load_and_scale_data(estimate_missing, scaling_method, split_point, seg_points)
    # Build P_initial matrix.
    P_initial, P_array = build_P_initial(full_df, interact_df)
    # Build K data.
    K_index, K_array, beta_counts = build_K_data(full_df, interact_df, estimate_missing)
    K_sparse, K_data, K_indices, K_indptr = convert_to_sparse(K_array)
    # Precompute mappings.
    unique_kinases, gene_kinase_counts, gene_alpha_starts, gene_kinase_idx, total_alpha, kinase_beta_counts, kinase_beta_starts = precompute_mappings(
        P_initial, K_index)
    # Initialize parameters.
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
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)
    post_optimization_results()
    optimization_performance()
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)
    logger.info(f'Report & Results {location(str(OUT_DIR))}')


if __name__ == "__main__":
    main()