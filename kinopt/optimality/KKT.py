import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from kinopt.evol.config.constants import OUT_FILE, OUT_DIR
from kinopt.evol.config.logconf import setup_logger

logger = setup_logger()


# -----------------------------
# LaTeX Table Generation
# -----------------------------
def generate_latex_table(summary_dict, table_caption, table=None):
    """
    Function to generate a LaTeX table from a summary dictionary.
    The table is formatted for use in a LaTeX document.

    :param summary_dict: Dictionary containing summary data.
    :param table_caption: Caption for the LaTeX table.
    :param table: Optional table object to format.
    :return: LaTeX table as a string.
    """
    latex_table = "\n\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|c|}\\hline\n"
    latex_table += "Metric & Value \\\\ \\hline\n"
    for key, value in summary_dict.items():
        latex_table += f"{key} & {value} \\\\ \\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += f"\\caption{{{table_caption}}}\n\\end{table}\n"
    return latex_table


# -----------------------------
# Printing Functions
# -----------------------------
def print_primal_feasibility_results(primal_summary, alpha_violations, beta_violations, logger_obj=None):
    """
    Logs the primal feasibility summary and violation details.
    """
    if logger_obj is None:
        logger_obj = logger
    logger_obj.info("Primal Feasibility Summary:")
    for key, value in primal_summary.items():
        logger_obj.info(f"{key}: {value}")

    logger_obj.info("Alpha Violations:")
    for index, value in alpha_violations.items():
        logger_obj.info(f"{index}: {value}")

    logger_obj.info("Beta Violations:")
    for index, value in beta_violations.items():
        logger_obj.info(f"{index}: {value}")

def print_sensitivity_and_active_constraints(sensitivity_summary, active_constraints_summary, logger_obj=None):
    """
    Logs the sensitivity summary and active constraints summary.
    """
    if logger_obj is None:
        logger_obj = logger
    logger_obj.info("Sensitivity Summary:")
    for key, value in sensitivity_summary.items():
        logger_obj.info(f"{key}: {value}")
    logger_obj.info("Active Constraints Summary:")
    for key, value in active_constraints_summary.items():
        logger_obj.info(f"{key}: {value}")


def plot_constraint_violations(alpha_violations, beta_violations, out_dir):
    """
    Function to plot constraint violations for alpha and beta values.
    It creates a stacked bar plot showing the violations for each protein.
    The top 5 proteins with the highest violations are highlighted in red.

    Args:
        alpha_violations (pd.Series): Series containing alpha constraint violations.
        beta_violations (pd.Series): Series containing beta constraint violations.
        out_dir (str): Directory to save the plot.
    """
    # Group and combine violations
    alpha_violations_abs = alpha_violations.abs().groupby('Gene').sum()
    beta_violations_abs = beta_violations.abs().reindex(alpha_violations_abs.index, fill_value=0)
    combined = pd.DataFrame({
        "Alpha Violations": alpha_violations_abs,
        "Beta Violations": beta_violations_abs
    })
    combined['Total Violations'] = combined.sum(axis=1)
    combined = combined.sort_values(by="Total Violations", ascending=True)
    top_proteins = combined.tail(5).index

    plt.figure(figsize=(8, 8))
    bar_alpha = plt.bar(combined.index, combined["Alpha Violations"], color='dodgerblue', label=r'$\alpha$')
    bar_beta = plt.bar(combined.index, combined["Beta Violations"], bottom=combined["Alpha Violations"],
                       color='lightgreen', label=r'$\beta$')

    # Highlight top violations in red
    for bar in bar_alpha:
        if bar.get_x() in top_proteins:
            bar.set_color('red')
    for bar in bar_beta:
        if bar.get_x() in top_proteins:
            bar.set_color('red')

    plt.xlabel("Proteins")
    plt.ylabel("Constraint Violations")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "constraint_violations.png"), dpi=300)
    plt.close()


def plot_sensitivity_analysis(sensitivity_analysis, out_dir):
    """
    Function to plot sensitivity analysis results.
    It creates a horizontal bar plot showing the mean, max, and min sensitivity for each protein.

    Args:
        sensitivity_analysis (pd.DataFrame): DataFrame containing sensitivity analysis results.
        out_dir (str): Directory to save the plot.
    """
    summary = sensitivity_analysis.groupby("GeneID")[["Sensitivity Mean", "Max Sensitivity", "Min Sensitivity"]].mean()
    summary = summary.sort_values(by="Sensitivity Mean", ascending=True)

    plt.figure(figsize=(8, 8))
    bar_min = plt.barh(summary.index, summary["Min Sensitivity"], color='lightgreen', label='Min')
    bar_mean = plt.barh(summary.index, summary["Sensitivity Mean"],
                        left=summary["Min Sensitivity"], color='dodgerblue', label='Mean')
    bar_max = plt.barh(summary.index, summary["Max Sensitivity"],
                       left=summary["Min Sensitivity"] + summary["Sensitivity Mean"],
                       color='coral', label='Max')
    plt.xlabel("Sensitivity")
    plt.ylabel("Proteins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "sensitivity.png"), dpi=300)
    plt.close()

def process_excel_results(file_path=OUT_FILE):
    """
    Function to process the Excel results file.
    It reads the alpha and beta values, estimated and observed values,
    validates normalization constraints, computes residuals and gradients,
    and generates LaTeX tables for the residuals and sensitivity summaries.
    It also performs sensitivity analysis and identifies high sensitivity sites.
    The results are returned as a dictionary.

    Args:
        file_path (str): Path to the Excel file containing results.
    Returns:
        dict: Dictionary containing the processed results, including alpha and beta values,
              estimated and observed values, constraint violations, residuals summary,
              sensitivity summary, and high sensitivity sites.
    """
    alpha_values = pd.read_excel(file_path, sheet_name='Alpha Values')
    beta_values = pd.read_excel(file_path, sheet_name='Beta Values')
    estimated_values = pd.read_excel(file_path, sheet_name='Estimated')
    observed_values = pd.read_excel(file_path, sheet_name='Observed')

    # Validate normalization constraints
    alpha_sum = alpha_values.groupby(['Gene', 'Psite'])['Alpha'].sum()
    alpha_violations = alpha_sum[alpha_sum != 1]
    beta_sum = beta_values.groupby(['Kinase'])['Beta'].sum()
    beta_violations = beta_sum[beta_sum != 1]

    # Compute residuals and gradients
    observed_matrix = observed_values.iloc[:, 2:].values
    estimated_matrix = estimated_values.iloc[:, 2:].values
    residuals = observed_matrix - estimated_matrix
    gradients = np.gradient(residuals, axis=1)
    residuals_summary = {
        "Max Residual": round(np.max(residuals), 2),
        "Min Residual": round(np.min(residuals), 2),
        "Mean Residual": round(np.mean(residuals), 2),
        "Max Gradient": round(np.max(gradients), 2),
        "Min Gradient": round(np.min(gradients), 2),
        "Mean Gradient": round(np.mean(gradients), 2)
    }
    sensitivity_summary = {
        "Max Sensitivity": round(np.max(observed_matrix), 2),
        "Min Sensitivity": round(np.min(observed_matrix), 2),
        "Mean Sensitivity": round(np.mean(observed_matrix), 2)
    }

    latex_res_table = generate_latex_table(residuals_summary, "Residual Summary")
    latex_sens_table = generate_latex_table(sensitivity_summary, "Sensitivity Summary")
    logger.info(latex_res_table)
    logger.info(latex_sens_table)

    sensitivity_analysis = pd.DataFrame({
        "GeneID": observed_values.iloc[:, 0],
        "Psite": observed_values.iloc[:, 1],
        "Sensitivity Mean": observed_matrix.mean(axis=1),
        "Max Sensitivity": np.max(observed_matrix, axis=1),
        "Min Sensitivity": np.min(observed_matrix, axis=1)
    })

    # Threshold for high sensitivity sites
    high_thresh = 0.75
    high_sites_idx = np.where(observed_matrix >= high_thresh)[0]
    high_sites = [
        (observed_values.iloc[i, 0], observed_values.iloc[i, 1])
        for i in high_sites_idx
    ]

    results = {
        "alpha_values": alpha_values,
        "beta_values": beta_values,
        "estimated_values": estimated_values,
        "observed_values": observed_values,
        "alpha_constraint_violations": alpha_violations,
        "beta_constraint_violations": beta_violations,
        "residuals_summary": residuals_summary,
        "sensitivity_summary": sensitivity_summary,
        "sensitivity_analysis": sensitivity_analysis,
        "high_sensitivity_sites": high_sites,
    }
    return results


def post_optimization_results():
    """
    Function to process and visualize the results of the optimization.
    It reads the results from an Excel file, processes the data,
    and generates plots for constraint violations and sensitivity analysis.
    It also prints the primal feasibility results and sensitivity summaries.
    The results are returned as a dictionary.

    Returns:
        dict: Dictionary containing the processed results, including alpha and beta values,
              estimated and observed values, constraint violations, residuals summary,
              sensitivity summary, and high sensitivity sites.
    """
    results = process_excel_results()
    # Plot violation and sensitivity figures
    plot_constraint_violations(results["alpha_constraint_violations"], results["beta_constraint_violations"], OUT_DIR)
    plot_sensitivity_analysis(results["sensitivity_analysis"], OUT_DIR)

    # For demonstration, call the printing functions (you may pass in appropriate summaries)
    print_primal_feasibility_results(results["alpha_constraint_violations"],
                                     results["alpha_constraint_violations"],
                                     results["beta_constraint_violations"])
    print_sensitivity_and_active_constraints(results["sensitivity_summary"],
                                             results["sensitivity_summary"])
    return results