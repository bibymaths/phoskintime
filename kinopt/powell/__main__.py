import shutil

from kinopt.evol.config.constants import OUT_FILE, ODE_DATA_DIR, OUT_DIR
from kinopt.optimality.KKT import post_optimization_results
from kinopt.powell.runpowell import run_powell
from config.helpers import location
from utils.display import create_report, organize_output_files
from kinopt.local.config.logconf import setup_logger
logger = setup_logger()

if __name__ == '__main__':
    """
    Main function to run the Powell optimization algorithm. 
    It sets up the environment, runs the optimization, and organizes the output files. 
    It also generates a report of the results.
    """
    # Set up the environment
    # Run the Powell optimization algorithm
    run_powell()
    # Copy the output file to the ODE_DATA_DIR
    shutil.copy(OUT_FILE, ODE_DATA_DIR / OUT_FILE.name)
    # Post-process the optimization results
    post_optimization_results()
    # Organize the output files
    organize_output_files(OUT_DIR)
    # Create a report of the results
    create_report(OUT_DIR)
    # Log the location of the report and results
    logger.info(f'Report & Results {location(str(OUT_DIR))}')