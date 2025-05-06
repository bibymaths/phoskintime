import shutil

from kinopt.evol.config import time_series_columns, METHOD
from kinopt.evol.config.constants import OUT_DIR, OUT_FILE, ODE_DATA_DIR
from kinopt.evol.config.helpers import location
from kinopt.evol.exporter.sheetutils import output_results
from kinopt.evol.objfn import estimated_series, residuals
from kinopt.evol.optcon.construct import check_kinases
from kinopt.fitanalysis import optimization_performance
from utils import latexit

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
        opt_analyze_nsga(problem, result, F, pairs, approx_ideal, approx_nadir, asf_i, pseudo_i, n_evals, hist_hv, hist,
                         val,
                         hist_cv_avg, k, hist_igd, best_objectives, waterfall_df, convergence_df, alpha_values,
                         beta_values)

    # Copy the output file to the ODE data directory.
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)

    # Perform post-optimization analysis.
    post_optimization_results()

    # Analyze the performance of the optimization.
    optimization_performance()

    # LateX the results
    latexit.main(OUT_DIR)

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
