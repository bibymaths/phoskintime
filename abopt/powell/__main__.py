import shutil

from abopt.evol.config.constants import OUT_FILE, ODE_DATA_DIR, OUT_DIR
from abopt.optimality.KKT import post_optimization_results
from abopt.powell.runpowell import run_powell
from config.helpers import location
from utils.display import create_report, organize_output_files
from abopt.local.config.logconf import setup_logger
logger = setup_logger()

if __name__ == '__main__':
    run_powell()
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)
    post_optimization_results()
    organize_output_files(OUT_DIR)
    create_report(OUT_DIR)
    logger.info(f'Report & Results {location(str(OUT_DIR))}')