
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from config.helpers import location
from config.config import parse_args, extract_config, log_config
from config.constants import model_type, OUT_DIR, TIME_POINTS, OUT_RESULTS_DIR, ESTIMATION_MODE
from config.logconf import setup_logger
from paramest.core import process_gene_wrapper
from utils.display import ensure_output_directory, save_result, organize_output_files, create_report

logger = setup_logger()
args = parse_args()
config = extract_config(args)
if config['profile_start'] is None or config['profile_end'] is None or config['profile_step'] is None:
    desired_times = None
else:
    desired_times = np.arange(
        config['profile_start'],
        config['profile_end'] + config['profile_step'],
        config['profile_step']
    )

def main():

    logger.info(f"{model_type} Phosphorylation Modelling Configuration - {ESTIMATION_MODE.upper()} Estimation Mode")
    log_config(logger, config['bounds'], config['fixed_params'], config['time_fixed'], args)
    ensure_output_directory(OUT_DIR)
    data = pd.read_excel(config['input_excel'], sheet_name='Estimated')
    genes = data["Gene"].unique().tolist()[:1]
    logger.info(f"Loaded Time Series for {len(genes)} Protein(s)")

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
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)
    logger.info(f'Report & Results {location(str(OUT_DIR))}')

if __name__ == "__main__":
    main()