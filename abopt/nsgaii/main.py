import shutil

from abopt.nsgaii.config import time_series_columns
from abopt.nsgaii.config.constants import OUT_DIR, OUT_FILE, ODE_DATA_DIR
from abopt.nsgaii.config.helpers import location
from abopt.nsgaii.exporter.plotout import opt_analyze
from abopt.nsgaii.exporter.sheetutils import output_results
from abopt.nsgaii.objfn import estimated_series, residuals
from abopt.nsgaii.objfn.minfn import PhosphorylationOptimizationProblem
from abopt.nsgaii.opt.optrun import run_optimization, post_optimization
from abopt.nsgaii.utils.iodata import organize_output_files, create_report
from abopt.nsgaii.optcon import P_initial, P_initial_array, K_array, K_index, beta_counts, gene_psite_counts
from abopt.nsgaii.utils.params import extract_parameters
from abopt.nsgaii.config.logconf import setup_logger
logger = setup_logger()


def main():
    problem, result = run_optimization(
        P_initial,
        P_initial_array,
        K_index,
        K_array,
        gene_psite_counts,
        beta_counts,
        PhosphorylationOptimizationProblem
    )

    (F, pairs, n_evals, hist_cv, hist_cv_avg, k, igd, hv, best_solution, best_objectives, optimized_params,
     approx_nadir, approx_ideal, scores, best_index, hist, hist_hv, hist_igd, convergence_df, waterfall_df, asf_i, pseudo_i,
     pairs, val) = post_optimization(result)

    alpha_values, beta_values = extract_parameters(P_initial, gene_psite_counts, K_index, best_solution.X)

    P_estimated = estimated_series(best_solution.X, P_initial, K_index, K_array, gene_psite_counts, beta_counts)

    res = residuals(P_initial_array, P_estimated)

    # Output results.
    output_results(P_initial, P_initial_array, P_estimated, res, alpha_values, beta_values,
                   result, time_series_columns, OUT_FILE)

    opt_analyze(problem, result, F, pairs, approx_ideal, approx_nadir, asf_i, pseudo_i, n_evals, hist_hv, hist, val,
                hist_cv_avg, k, hist_igd, best_objectives, waterfall_df, convergence_df, alpha_values, beta_values)
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)
    logger.info(f'Report & Results {location(str(OUT_DIR))}')

if __name__ == "__main__":
    main()