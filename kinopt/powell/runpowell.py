import subprocess
import os
from kinopt.evol.config.logconf import setup_logger
logger = setup_logger(name="phoskintime")

def run_powell():
    """
    Run the Powell optimization algorithm using Julia.
    It sets the number of threads to half of the available threads
    and logs the output in real time.
    """
    # Run the command to get the number of threads (as a string)
    result = subprocess.run("lscpu -p | grep -v '^#' | wc -l", shell=True, capture_output=True, text=True)
    num_threads_str = result.stdout.strip()

    # Convert to integer, compute half, and round it
    num_threads_int = int(num_threads_str)
    half_threads = round(num_threads_int / 2)

    # Set up the environment variable (convert back to string)
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = str(half_threads)

    # Start the Julia script and log output in real time
    process = subprocess.Popen(
        ["julia", "kinopt/powell/powell.jl"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1
    )

    # Read each line
    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            logger.info(line.rstrip())

    process.wait()
