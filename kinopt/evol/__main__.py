import shutil

from kinopt.evol.config import time_series_columns, METHOD
from kinopt.evol.config.constants import OUT_DIR, OUT_FILE, ODE_DATA_DIR
from kinopt.evol.config.helpers import location
from kinopt.evol.exporter.sheetutils import output_results
from kinopt.evol.objfn import estimated_series, residuals
from kinopt.evol.optcon.construct import check_kinases
from kinopt.fitanalysis import optimization_performance

if METHOD == "DE":
    from kinopt.evol.objfn.minfndiffevo import PhosphorylationOptimizationProblem
    from kinopt.evol.opt.optrun import run_optimization, post_optimization_de
    from kinopt.evol.exporter.plotout import opt_analyze_de
else:
    from kinopt.evol.objfn.minfnnsgaii import PhosphorylationOptimizationProblem
    from kinopt.evol.opt.optrun import run_optimization, post_optimization_nsga
    from kinopt.evol.exporter.plotout import opt_analyze_nsga
from kinopt.evol.utils.iodata import organize_output_files, create_report
from kinopt.evol.optcon import P_initial, P_initial_array, K_array, K_index, beta_counts, gene_psite_counts
from kinopt.evol.utils.params import extract_parameters
from kinopt.evol.config.logconf import setup_logger
from kinopt.optimality.KKT import post_optimization_results

logger = setup_logger()

def main():
    """
    Main function to run the optimization process.

    It initializes the optimization problem, runs the optimization,
    and processes the results.

    The function performs the following steps:
    1. Initializes the optimization problem using the provided parameters.
    2. Runs the optimization algorithm (either DE or NSGA-II).
    3. Processes the results of the optimization.
    4. Outputs the results to an Excel file.
    5. Analyzes the optimization results.
    6. Copies the output file to a specified directory.
    7. Generates a report and organizes output files.
    8. Logs the completion of the process.

    The function takes no parameters and returns nothing.

    The function uses the following global variables:
    - P_initial: Initial mapping of gene-psite pairs to kinase relationships and time-series data.
    - P_initial_array: Array containing observed time-series data for gene-psite pairs.
    - K_array: Array containing time-series data for kinase-psite combinations.
    - K_index: Mapping of kinases to their respective psite data.
    - beta_counts: Mapping of kinase indices to the number of associated psites.
    - gene_psite_counts: Number of kinases per gene-psite combination.
    - OUT_DIR: Directory for output files.
    - OUT_FILE: Name of the output file.
    - ODE_DATA_DIR: Directory for ODE data files.
    - time_series_columns: List of time series columns to extract.

    The function uses the following helper functions:
    - run_optimization: Runs the optimization algorithm.
    - post_optimization_de: Processes the results of the DE optimization.
    - post_optimization_nsga: Processes the results of the NSGA-II optimization.
    - extract_parameters: Extracts parameters from the optimization results.
    - estimated_series: Estimates the series based on the optimization results.
    - residuals: Computes the residuals between the observed and estimated series.
    - output_results: Outputs the results to an Excel file.
    - opt_analyze_de: Analyzes the DE optimization results.
    - opt_analyze_nsga: Analyzes the NSGA-II optimization results.
    - post_optimization_results: Performs post-optimization analysis.
    - optimization_performance: Analyzes the performance of the optimization.
    - organize_output_files: Organizes the output files.
    - create_report: Generates a report based on the optimization results.

    The function uses the following libraries:
    - shutil: For file operations.
    - kinopt.evol.config: For configuration settings.
    - kinopt.evol.exporter: For exporting results.
    - kinopt.evol.objfn: For defining the optimization problem.
    - kinopt.evol.opt: For running the optimization.
    - kinopt.evol.utils: For utility functions.
    - kinopt.optimality: For optimality analysis.
    - kinopt.evol.config.logconf: For logging configuration.
    - kinopt.evol.config.helpers: For helper functions.
    """
    logger.info('[Global Optimization] Started - Kinase Phosphorylation Optimization Problem')

    # Check for the missing kinases in the input files.
    # From the input2.csv file, it checks if the kinases are present in the input1.csv file.
    check_kinases()

    # Initialize the optimization problem.
    problem, result = run_optimization(
        P_initial,
        P_initial_array,
        K_index,
        K_array,
        gene_psite_counts,
        beta_counts,
        PhosphorylationOptimizationProblem
    )

    # Run the optimization algorithm.
    if METHOD == "DE":
        alpha_values, beta_values = extract_parameters(P_initial, gene_psite_counts, K_index, result.X)
        (ordered_optimizer_runs, convergence_df,
         long_df, x_values, y_values, val) = post_optimization_de(result, alpha_values, beta_values)
        P_estimated = estimated_series(result.X, P_initial, K_index, K_array, gene_psite_counts, beta_counts)
    else:
        (F, pairs, n_evals, hist_cv, hist_cv_avg, k, igd, hv, best_solution, best_objectives, optimized_params,
         approx_nadir, approx_ideal, scores, best_index, hist, hist_hv, hist_igd, convergence_df, waterfall_df, asf_i,
         pseudo_i,
         pairs, val) = post_optimization_nsga(result)
        alpha_values, beta_values = extract_parameters(P_initial, gene_psite_counts, K_index, best_solution.X)
        P_estimated = estimated_series(best_solution.X, P_initial, K_index, K_array, gene_psite_counts, beta_counts)

    # Compute residuals.
    res = residuals(P_initial_array, P_estimated)

    # Output results.
    output_results(P_initial, P_initial_array, P_estimated, res, alpha_values, beta_values,
                   result, time_series_columns, OUT_FILE)

    # Analyze the optimization results.
    if METHOD == "DE":
        opt_analyze_de(long_df, convergence_df, ordered_optimizer_runs, x_values, y_values, val)
    else:
        opt_analyze_nsga(problem, result, F, pairs, approx_ideal, approx_nadir, asf_i, pseudo_i, n_evals, hist_hv, hist, val,
                    hist_cv_avg, k, hist_igd, best_objectives, waterfall_df, convergence_df, alpha_values, beta_values)

    # Copy the output file to the ODE data directory.
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)

    # Perform post-optimization analysis.
    post_optimization_results()

    # Analyze the performance of the optimization.
    optimization_performance()

    # Organize the output files.
    organize_output_files(OUT_DIR)

    # Create a report.
    create_report(OUT_DIR)

    # Log the completion of the process.
    logger.info(f'Report & Results {location(str(OUT_DIR))}')

    # Click to open the report in a web browser.
    for fpath in [OUT_DIR / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

if __name__ == "__main__":
    main()