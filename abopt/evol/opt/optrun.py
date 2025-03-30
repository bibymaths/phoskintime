import io
import contextlib
import numpy as np
import subprocess
import pandas as pd
from multiprocessing.pool import ThreadPool

from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import Hypervolume
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from abopt.evol.config import METHOD
from abopt.evol.config.constants import OUT_DIR
from abopt.evol.config.logconf import setup_logger
logger = setup_logger()

def run_optimization(
    P_initial,
    P_initial_array,
    K_index,
    K_array,
    gene_psite_counts,
    beta_counts,
    PhosphorylationOptimizationProblem
):
    """
    Sets up and runs the multi-objective optimization problem for phosphorylation
    using an NSGA2 algorithm and a thread pool for parallelization.

    Args:
        P_initial, P_initial_array, K_index, K_array, gene_psite_counts, beta_counts:
            Data structures describing the problem (time-series data, kinases, etc.).
        PhosphorylationOptimizationProblem (class):
            The custom problem class to be instantiated.

    Returns:
        result: The pymoo result object containing the optimized population and history.
        exec_time: Execution time for the optimization.
    """
    # 1) Determine the number of threads via lscpu
    n_threads_cmd = "lscpu -p | grep -v '^#' | wc -l"
    n_threads = int(subprocess.check_output(n_threads_cmd, shell=True).decode().strip())
    logger.info(f"Number of threads: {n_threads}")

    # 2) Create a thread pool and a parallelization runner
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # 3) Instantiate the problem
    problem = PhosphorylationOptimizationProblem(
        P_initial=P_initial,
        P_initial_array=P_initial_array,
        K_index=K_index,
        K_array=K_array,
        gene_psite_counts=gene_psite_counts,
        beta_counts=beta_counts,
        elementwise_runner=runner
    )
    if METHOD == "DE":
        algorithm = DE()
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-12,
            cvtol=1e-12,
            ftol=1e-12,
            period=20,
            n_max_gen=1000,
            n_max_evals=100000
        )
    else:
        algorithm = NSGA2(
        pop_size=100,
        crossover=TwoPointCrossover(),
        eliminate_duplicates=True)

        termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        n_max_gen=1000,
        n_max_evals=100000)

    # 5) Run the optimization
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = minimize(
            problem,
            algorithm,
            termination=termination,
            verbose=True,
            save_history=True
        )

    # Log the captured pymoo progress
    pymoo_progress = buf.getvalue()
    if pymoo_progress.strip():  # only log if there's actual text
        logger.info("--- NSGA-II Progress Output ---\n" + pymoo_progress)

    # 6) Grab execution time and close the pool
    # Convert execution time to minutes and hours
    exec_time_seconds = result.exec_time
    exec_time_minutes = exec_time_seconds / 60
    exec_time_hours = exec_time_seconds / 3600
    logger.info(f"Execution Time: {exec_time_seconds:.2f} seconds |  "
                f"{exec_time_minutes:.2f} minutes |  "
                f"{exec_time_hours:.2f} hours")
    pool.close()

    return problem, result

def post_optimization(
    result,
    weights=np.array([1.0, 1.0, 1.0]),
    ref_point=np.array([3, 1, 1])):
    """
    Post-processes the result of a multi-objective optimization run.
    1) Extracts the Pareto front and computes a weighted score to pick a 'best' solution.
    2) Gathers metrics like Hypervolume (HV) and IGD+ over the optimization history.
    3) Logs feasibility generation info, saves waterfall and convergence data to CSV.

    Args:
        result: The final result object from the optimizer (e.g., a pymoo result).
        weights (np.ndarray): Array of length 3 for weighting the objectives.
        ref_point (np.ndarray): Reference point for hypervolume computations.

    Returns:
        dict: A dictionary with keys:
            'best_solution': The best individual from the weighted scoring.
            'best_objectives': Its corresponding objective vector.
            'optimized_params': The individual's decision variables (X).
            'scores': Weighted scores for each solution in the Pareto front.
            'best_index': The index of the best solution according to weighted score.
            'hist_hv': The hypervolume per generation.
            'hist_igd': The IGD+ per generation.
            'convergence_df': The DataFrame with iteration vs. best objective value
                for each iteration in the result history.
    """
    # 1) Extract the Pareto front from the final population
    pareto_front = np.array([ind.F for ind in result.pop])

    # Weighted scoring for picking a single 'best' solution from Pareto
    scores = pareto_front[:, 0] + weights[1]*np.abs(pareto_front[:, 1]) + weights[2]*np.abs(pareto_front[:, 2])
    best_index = np.argmin(scores)
    best_solution = result.pop[best_index]
    best_objectives = pareto_front[best_index]
    optimized_params = best_solution.X

    # Additional references from the result
    F = result.F  # The entire final objective set
    pairs = [(0,1),(0,2),(1,2)]
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    # Decomposition objects
    decomp = ASF()
    asf_i = decomp.do(F, 1/weights).argmin()

    hist = result.history  # the full history of the optimization

    n_evals = []
    hist_F = []
    hist_cv = []
    hist_cv_avg = []

    # 2) Gather feasibility, objective space data over each generation
    for algo in hist:
        n_evals.append(algo.evaluator.n_eval)

        opt = algo.opt
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    # Identify when we first got a feasible solution
    k = np.where(np.array(hist_cv) <= 0.0)[0].min()
    logger.info(f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations.")

    # Identify when the whole population became feasible
    vals = hist_cv_avg
    k = np.where(np.array(vals) <= 0.0)[0].min()
    logger.info(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

    # 3) Metrics: Hypervolume & IGD+ across generations
    metric_hv = Hypervolume(
        ref_point=ref_point,
        norm_ref_point=False,
        zero_to_one=True,
        ideal=approx_ideal,
        nadir=approx_nadir
    )
    hist_hv = [metric_hv.do(_F) for _F in hist_F]

    metric_igd = IGDPlus(F)
    hist_igd = [metric_igd.do(_F) for _F in hist_F]

    # 4) Waterfall or 'pops' data (currently empty in snippet)
    pops = []
    # You could populate 'pops' with relevant info from each generation if desired

    waterfall_df = pd.DataFrame(pops)
    waterfall_df.to_csv(f'{OUT_DIR}/parameter_scan.csv', index=False)

    # 5) Convergence data (best objective value each generation)
    val = [e.opt.get("F")[0] for e in hist]  # each iteration's best F
    # Flatten if it is list or array
    flattened_val = [
        v[0] if isinstance(v, (list, np.ndarray)) else v
        for v in val
    ]
    convergence_df = pd.DataFrame({
        "Iteration": np.arange(len(flattened_val)),
        "Value": flattened_val
    })
    convergence_df.to_csv(f'{OUT_DIR}/convergence.csv', index=False)

    # Display the selected solution and objectives
    logger.info("--- Best Solution ---")
    logger.info(f"Objective Values (F): {best_objectives}")

    return (
        F,
        pairs,
        n_evals,
        hist_cv,
        hist_cv_avg,
        k,
        metric_igd,
        metric_hv,
        best_solution,
        best_objectives,
        optimized_params,
        approx_nadir,
        approx_ideal,
        scores,
        best_index,
        hist,
        hist_hv,
        hist_igd,
        convergence_df,
        waterfall_df,
        asf_i,
        PseudoWeights(weights).do(F),
        pairs,
        val
    )
