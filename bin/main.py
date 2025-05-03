import pandas as pd
from tqdm import tqdm
from config.helpers import location
from config.config import parse_args, extract_config, log_config
from config.constants import (model_type, OUT_DIR, TIME_POINTS, OUT_RESULTS_DIR, DEV_TEST,
                              NUM_TRAJECTORIES, PARAMETER_SPACE, PERTURBATIONS_VALUE, ALPHA_CI,
                              SENSITIVITY_ANALYSIS, USE_REGULARIZATION, Y_METRIC, Y_METRIC_DESCRIPTIONS,
                              DELTA_WEIGHT, ALPHA_WEIGHT, BETA_WEIGHT, GAMMA_WEIGHT, MU_WEIGHT)
from config.logconf import setup_logger
from paramest.core import process_gene_wrapper
from plotting import Plotter
from utils import latexit
from utils.display import ensure_output_directory, save_result, organize_output_files, create_report, merge_obs_est

logger = setup_logger()

# Check if OUT_DIR, TIME_POINTS, OUT_RESULTS_DIR, ESTIMATION_MODE are defined
if OUT_DIR is None or TIME_POINTS is None or OUT_RESULTS_DIR is None:
    logger.error("Output directory, time points, not defined. Exiting.")
    exit(1)

# Parse command line arguments and extract configuration
args = parse_args()

# Check if the arguments are valid
if not args:
    logger.error("Invalid arguments. Exiting.")
    exit(1)
config = extract_config(args)

# Check if the configuration is valid
if not config:
    logger.error("Invalid configuration. Exiting.")
    exit(1)

def main():
    """
    Main function to run the phosphorylation modelling process.
    It reads the configuration, loads the data, and processes each gene in parallel.
    It also handles logging and output organization.
    """
    # Set up the logger
    logger.info("           --------------------------------")
    logger.info(f"{model_type} Phosphorylation Modelling Configuration")
    logger.info("           --------------------------------")
    logger.info(f"      i = Number of phosphorylation sites (Residue_Position) in the model")
    logger.info(f"      Confidence Interval: {ALPHA_CI*100}")
    logger.info(f"      Sensitivity Analysis: {SENSITIVITY_ANALYSIS}")
    logger.info(f"      - Metric: {' '.join(part.upper() for part in Y_METRIC.split('_'))}")
    logger.info(f"      - {Y_METRIC_DESCRIPTIONS.get(Y_METRIC, "No description available.")}")
    logger.info(f"      - Number of Trajectories: {NUM_TRAJECTORIES}")
    logger.info(f"      - Parameter Space: {PARAMETER_SPACE}")
    logger.info(f"      - Perturbations: {PERTURBATIONS_VALUE}")
    log_config(logger, config['bounds'], args)

    # Make output directory
    ensure_output_directory(OUT_DIR)

    # Load the data
    kinase_data = pd.read_excel(config['input_excel'], sheet_name='Estimated')
    mrna_data = pd.read_excel(config['input_excel_rna'], sheet_name='Estimated')

    # Check if the data is empty
    if mrna_data.empty and kinase_data.empty:
        logger.error("No data found in the input Excel files.")
        return

    # Check if the required columns are present: Gene, Psite, x1 - x14
    required_columns = ['Gene', 'Psite'] + [f'x{i}' for i in range(1, 15)]
    missing_columns = [col for col in required_columns if col not in kinase_data.columns]
    if missing_columns:
        logger.error(f"Missing columns in the phosphorylation data: {', '.join(missing_columns)}")
        return

    # Check if the required columns are present in mRNA data: mRNA, x1 - x9
    required_mrna_columns = ['mRNA'] + [f'x{i}' for i in range(1, 10)]
    missing_mrna_columns = [col for col in required_mrna_columns if col not in mrna_data.columns]
    if missing_mrna_columns:
        logger.error(f"Missing columns in the mRNA data: {', '.join(missing_mrna_columns)}")
        return

    # Extract unique values from both datasets
    proteins = set(kinase_data['Gene'].dropna().unique())
    mrnas = set(mrna_data['mRNA'].dropna().unique())

    # Get sorted common proteins
    common_proteins = sorted(proteins.intersection(mrnas))

    # Get sorted non-common proteins
    non_common = sorted(proteins.symmetric_difference(mrnas))

    if not common_proteins:
        logger.warning("No common proteins found between phosphorylation and mRNA data.")
    else:
        logger.info(f"Genes found in phosphorylation data: {len(proteins)}")
        logger.info("  " + " ".join(f"[{gene}]" for gene in proteins))
        logger.info(f"Genes found in mRNA data: {len(mrnas)}")
        logger.info("  " + " ".join(f"[{rna}]" for rna in mrnas))
        logger.info(f"Genes found common between phosphorylation and mRNA data: {len(common_proteins)}")
        logger.info("  " + " ".join(f"[{gene}]" for gene in common_proteins))
        logger.info(f"Genes NOT found in both datasets: {len(non_common)}")
        logger.info("  " + " ".join(f"[{gene}]" for gene in non_common))
        logger.info("           --------------------------------")

    if DEV_TEST:
        # Load only gene 'X'
        _test = "ABL2"
        if _test in kinase_data["Gene"].values:
            genes = kinase_data[kinase_data["Gene"] == _test]["Gene"].unique().tolist()
        else:
            raise ValueError(f"{_test} not found in the input data.")
    else:
        # Load all protein groups
        genes = common_proteins

    # Check if the genes are empty
    if not genes:
        logger.error("No genes found in the input data.")
        return

    results = []
    for gene in genes:
        logger.info(f"[{gene}]      Processing...")
        result = process_gene_wrapper(
            gene, kinase_data, mrna_data, TIME_POINTS,
            config['bounds'], config['bootstraps']
        )
        results.append(result)

    # Check if the results are empty
    if not results:
        logger.error("No results found after processing.")
        return

    # Save the results
    save_result(results, excel_filename=OUT_RESULTS_DIR)

    # Merge the observed data and model fits for each gene
    merged_df = merge_obs_est(OUT_RESULTS_DIR)

    # Plot goodness of fit.
    Plotter("", OUT_DIR).plot_gof(merged_df)

    # Plot Kullback-Leibler divergence.
    Plotter("", OUT_DIR).plot_kld(merged_df)

    # Plot parameter relationships - profiles
    Plotter("", OUT_DIR).plot_top_param_pairs(OUT_RESULTS_DIR)

    # Plot regularization term values.
    Plotter("", OUT_DIR).plot_regularization(OUT_RESULTS_DIR)

    # Plot model protein wise errors.
    Plotter("", OUT_DIR).plot_model_error(OUT_RESULTS_DIR)

    logger.info("Plotting completed.")

    # LateX the results
    latexit.main(OUT_DIR)

    logger.info(f"LateX generated.")

    # Organize output files and create a report
    organize_output_files([OUT_DIR])
    create_report(OUT_DIR)

    logger.info("           --------------------------------")
    logger.info(f"Report & Results {location(str(OUT_DIR))}")

    # Click to open the report in a web browser.
    for fpath in [OUT_DIR / 'report.html']:
        logger.info(f"{fpath.as_uri()}")

    logger.info("           --------------------------------")

if __name__ == "__main__":
    main()
