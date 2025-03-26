
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.constants import OUT_DIR
from estimation.core import process_gene_wrapper
from utils.io_utils import ensure_output_directory, save_result
from config.logging_config import setup_logger
from config.config import parse_args, extract_config, log_config
logger = setup_logger(__name__)

def main():
    args = parse_args()
    config = extract_config(args)
    logger.info("Parsed arguments and loaded configuration.")
    log_config(logger, config['bounds'], config['fixed_params'], config['time_fixed'], args)
    ensure_output_directory(OUT_DIR)
    logger.info(f"Ensured output directory exists at: {OUT_DIR}")
    time_points = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])
    desired_times = np.arange(config['profile_start'], config['profile_end'] + config['profile_step'], config['profile_step'])
    measurement_data = pd.read_excel(config['input_excel'], sheet_name='Estimated Values')
    genes = measurement_data["Gene"].unique().tolist()[:1]
    logger.info(f"Loaded measurement data with {len(genes)} genes.")

    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        results = list(tqdm(
            executor.map(process_gene_wrapper, genes,
                         [measurement_data] * len(genes),
                         [time_points] * len(genes),
                         [config['bounds']] * len(genes),
                         [config['fixed_params']] * len(genes),
                         [desired_times] * len(genes),
                         [config['time_fixed']] * len(genes),
                         [config['bootstraps']] * len(genes)),
            total=len(genes),
            desc="Processing Genes"
        ))

    save_result(results, time_points, )
    logger.info("All results saved successfully.")

if __name__ == "__main__":
    main()