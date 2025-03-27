
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from config.config import parse_args, extract_config, log_config
from config.constants import OUT_DIR, TIME_POINTS, OUT_RESULTS_DIR
from config.logging_config import setup_logger
from estimation.core import process_gene_wrapper
from utils.utils import ensure_output_directory, save_result, organize_output_files

logger = setup_logger(__name__)

def main():

    args = parse_args()
    config = extract_config(args)

    logger.info("Distributive Phosphorylation Modelling Configuration - Time Profiles")
    logger.info("Parsed arguments and loaded configuration.")

    log_config(logger, config['bounds'], config['fixed_params'], config['time_fixed'], args)

    ensure_output_directory(OUT_DIR)
    logger.info(f"Ensured output directory exists at: {OUT_DIR}")

    desired_times = np.arange(config['profile_start'], config['profile_end'] + config['profile_step'], config['profile_step'])
    data = pd.read_excel(config['input_excel'], sheet_name='Estimated Values')
    genes = data["Gene"].unique().tolist()[:1]
    logger.info(f"Loaded Time Series for {len(genes)} Protein(s).")

    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        results = list(executor.map(
            process_gene_wrapper, genes,
            [data] * len(genes),
            [TIME_POINTS] * len(genes),
            [config['bounds']] * len(genes),
            [config['fixed_params']] * len(genes),
            [desired_times] * len(genes),
            [config['time_fixed']] * len(genes),
            [config['bootstraps']] * len(genes)
        ))

    save_result(results, excel_filename=OUT_RESULTS_DIR)

    logger.info("Parameter Estimation Finished")
    logger.info(f"Plots Saved at {OUT_DIR}")

    organize_output_files(OUT_DIR)
    logger.info("All results saved successfully.")

if __name__ == "__main__":
    main()