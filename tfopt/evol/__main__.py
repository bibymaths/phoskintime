import shutil

from kinopt.evol.config.helpers import location
from tfopt.evol.config.constants import parse_args, OUT_DIR, OUT_FILE, ODE_DATA_DIR
from tfopt.evol.exporter import post_processing
from tfopt.evol.objfn.minfn import TFOptimizationMultiObjectiveProblem
from tfopt.evol.opt.optrun import run_optimization
from tfopt.evol.optcon.construct import build_fixed_arrays
from tfopt.evol.config.logconf import setup_logger
from tfopt.evol.optcon.filter import filter_mrna, update_regulations, filter_TF, load_raw_data, determine_T_use
from tfopt.evol.utils.iodata import organize_output_files, create_report
from tfopt.evol.utils.params import create_no_psite_array, compute_beta_indices, create_initial_guess, create_bounds, \
    get_parallel_runner, print_alpha_mapping, print_beta_mapping, extract_best_solution
from tfopt.fitanalysis.helper import Plotter

logger = setup_logger()

# -------------------------------
# Main Routine
# -------------------------------
def main():
    """
    Main function to run the mRNA-TF optimization problem.

    This function performs the following steps:
    1. Parse command line arguments.
    2. Load raw input data (mRNA, TF, and regulation maps).
    3. Filter mRNA and TF data based on regulations.
    4. Determine common time series length.
    5. Build fixed shape arrays for optimization.
    6. Create initial guess vector and define bounds.
    7. Setup parallel runner for optimization.
    8. Create multi-objective optimization problem instance.
    9. Run the optimization using the specified optimizer.
    10. Extract the best solution and display mappings.
    11. Perform post-processing (prediction, plotting, and Excel output).
    12. Generate plots and organize output files.
    13. Create a report of the results.
    14. Log the completion of the optimization process.
    """
    logger.info("[Global Optimization] mRNA-TF Optimization Problem started")

    # Parse command line arguments.
    lb, ub, loss_type, optimizer = parse_args()

    # Load raw input data.
    (mRNA_ids, mRNA_mat, mRNA_time_cols,
     TF_ids, protein_dict, psite_dict, psite_labels_dict, TF_time_cols, reg_map) = load_raw_data()

    # Filter mRNA and update regulations.
    mRNA_ids, mRNA_mat = filter_mrna(mRNA_ids, mRNA_mat, reg_map)
    relevant_TFs = update_regulations(mRNA_ids, reg_map, TF_ids)
    TF_ids, protein_dict, psite_dict, psite_labels_dict = filter_TF(TF_ids, protein_dict,
                                                                    psite_dict, psite_labels_dict,
                                                                    relevant_TFs)

    # logger.info(f"Names of mRNAs: {mRNA_ids}")
    # logger.info(f"Names of TFs: {TF_ids}")
    # logger.info(f"Names of TFProtein: {protein_dict}")
    # logger.info(f"Names of TFPsiteData: {psite_dict}")
    # logger.info(f"Names of TFPsiteLabels: {psite_labels_dict}")
    # logger.info(f"Names of TFTimeCols: {TF_time_cols}")
    # logger.info(f"Names of RegMap: {reg_map}")

    # Determine the common time series length.
    T_use = determine_T_use(mRNA_mat, TF_time_cols)
    mRNA_mat = mRNA_mat[:, :T_use]

    # Build fixed shape arrays.
    (mRNA_mat, regulators, protein_mat, psite_tensor, n_reg,
     n_psite_max, psite_labels_arr, num_psites) = build_fixed_arrays(
        mRNA_ids, mRNA_mat, TF_ids, protein_dict, psite_dict, psite_labels_dict, reg_map)

    n_mRNA = mRNA_mat.shape[0]
    n_TF = protein_mat.shape[0]

    # logger.info(f"Number of mRNAs: {n_mRNA}")
    # logger.info(f"Number of TFs: {n_TF}")

    # Create an array marking TFs with no PSite data.
    no_psite_tf = create_no_psite_array(n_TF, num_psites, psite_labels_arr)

    # Compute cumulative starting indices for beta parameters.
    beta_start_indices, n_beta_total = compute_beta_indices(num_psites, n_TF)

    # Build the initial guess vector for optimization.
    x0, n_alpha = create_initial_guess(n_mRNA, n_reg, n_TF, num_psites, no_psite_tf)

    # Create bounds for the optimization variables.
    xl, xu = create_bounds(n_alpha, n_beta_total, lb, ub)
    total_dim = len(x0)

    # Setup parallel runner.
    runner, pool = get_parallel_runner()

    # Create the multi-objective problem instance.
    problem = TFOptimizationMultiObjectiveProblem(
        n_var=total_dim, n_mRNA=n_mRNA, n_TF=n_TF, n_reg=n_reg,
        n_alpha=n_alpha, mRNA_mat=mRNA_mat, regulators=regulators,
        protein_mat=protein_mat, psite_tensor=psite_tensor, T_use=T_use,
        no_psite_tf=no_psite_tf, xl=xl, xu=xu, beta_start_indices=beta_start_indices,
        num_psites=num_psites, n_psite_max=n_psite_max,
        loss_type=loss_type,
        elementwise_runner=runner
    )

    # Run the optimization.
    res = run_optimization(problem, total_dim, optimizer)

    if res.X is None:
        logger.info("No feasible solution found by pymoo. Exiting.")
        pool.close()
        return
    pool.close()

    # Extract the best solution and display mappings.
    final_alpha, final_beta, best_objectives, final_x = extract_best_solution(
        res, n_alpha, n_mRNA, n_reg, n_TF, num_psites, beta_start_indices)

    logger.info("--- Best Solution ---")
    logger.info(f"Objective Values (F): {best_objectives}")

    # Display the mappings of alpha and beta parameters.
    print_alpha_mapping(mRNA_ids, reg_map, TF_ids, final_alpha)
    print_beta_mapping(TF_ids, final_beta, psite_labels_arr)

    # Perform post-processing.
    post_processing(final_x, regulators, protein_mat, psite_tensor, n_reg, n_mRNA, T_use, n_mRNA,
                    beta_start_indices, num_psites, mRNA_ids, mRNA_mat, mRNA_time_cols, TF_ids,
                    final_alpha, final_beta, psite_labels_arr, best_objectives, reg_map)

    # Generate plots
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

    # Copy result file to ODE_DATA_DIR
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)

    # Organize output files and create a report.
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)

    logger.info(f'[Global] Report & Results {location(str(OUT_DIR))}')

    # Click to open the report in a web browser.
    for fpath in [OUT_DIR / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

if __name__ == "__main__":
    main()