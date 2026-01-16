import signal
import time

import numpy as np
import subprocess
import pandas as pd
from multiprocessing.pool import ThreadPool

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.indicators.hv import Hypervolume
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.problem import StarmapParallelization
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination

from kinopt.evol.config import METHOD
from kinopt.evol.config.constants import OUT_DIR
from kinopt.evol.config.logconf import setup_logger

logger = setup_logger()

def _run_with_ctrlc(problem, algorithm, termination, verbose=True, save_history=True):
    """
    Runs pymoo algorithm in a manual loop so Ctrl+C returns best-so-far safely.
    """
    # Optional: make Ctrl+C deterministic (convert SIGINT into KeyboardInterrupt consistently)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    t0 = time.perf_counter()

    # IMPORTANT: setup happens on the same algorithm instance you will finalize later
    algorithm.setup(
        problem,
        termination=termination,
        verbose=verbose,
        save_history=save_history
    )

    interrupted = False

    try:
        while algorithm.has_next():
            algorithm.next()
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Optimization interrupted by user (Ctrl+C). Finalizing best-so-far...")
    finally:
        res = algorithm.result()
        # Pymoo may not set exec_time in this path; set it yourself
        res.exec_time = time.perf_counter() - t0
        res.interrupted = interrupted

    return res

def choose_de_pop_size(problem):
    n_var = int(problem.n_var)

    pop = 10 * n_var
    pop = max(100, pop)
    pop = min(600, pop)

    # DE benefits from even population sizes
    pop = int(10 * round(pop / 10))
    return pop

def choose_nsga_pop_size(problem, n_obj=3):
    n_var = int(problem.n_var)

    if n_var <= 50:
        pop = 200
    elif n_var <= 150:
        pop = 400
    elif n_var <= 400:
        pop = 600
    else:
        pop = 800

    pop = int(50 * round(pop / 50))
    pop = max(pop, 10 * n_obj)  # never below 30 for 3 objectives
    return pop

def binary_tournament_loss_cv(pop, P, eps_cv=1e-10, cv_mode="linf", **kwargs):
    """
    Robust binary tournament comparator.

    Works for:
      A) single-objective: F has length 1
         - if CV exists, use constraint-domination (CV first, then F)
         - else compare by F only
      B) pseudo-constrained objectives: F = [loss, alpha_violation, beta_violation]
         - feasibility-first based on F[1], F[2], then loss
    """
    n_tournaments, n_competitors = P.shape
    if n_competitors != 2:
        raise ValueError("Only pressure=2 allowed for binary tournament!")

    S = np.full(n_tournaments, -1, dtype=int)

    def agg_cv_from_two(F_row):
        cv1 = abs(float(F_row[1]))
        cv2 = abs(float(F_row[2]))
        if cv_mode == "linf":
            return max(cv1, cv2)
        elif cv_mode == "l1":
            return cv1 + cv2
        elif cv_mode == "l2":
            return (cv1 * cv1 + cv2 * cv2) ** 0.5
        else:
            raise ValueError("cv_mode must be 'linf', 'l1', or 'l2'")

    def get_cv(ind):
        # Prefer true constraint violation if present (pymoo sets CV when n_ieq_constr>0)
        cv = getattr(ind, "CV", None)
        if cv is None:
            return None
        cv = float(np.asarray(cv).reshape(-1)[0])
        if not np.isfinite(cv):
            return np.inf
        return cv

    for i in range(n_tournaments):
        a, b = P[i]
        ia, ib = pop[a], pop[b]

        Fa = np.asarray(ia.F, dtype=float).reshape(-1)
        Fb = np.asarray(ib.F, dtype=float).reshape(-1)

        # Defensive: treat non-finite objective as worst
        if not np.all(np.isfinite(Fa)) and np.all(np.isfinite(Fb)):
            S[i] = b
            continue
        if not np.all(np.isfinite(Fb)) and np.all(np.isfinite(Fa)):
            S[i] = a
            continue

        # --- Case A: single-objective (typical GA/DE/PSO setup)
        if Fa.size == 1 and Fb.size == 1:
            cva = get_cv(ia)
            cvb = get_cv(ib)

            # If CV exists, use constraint-domination
            if cva is not None and cvb is not None:
                a_feas = cva <= eps_cv
                b_feas = cvb <= eps_cv

                if a_feas and not b_feas:
                    S[i] = a
                    continue
                if b_feas and not a_feas:
                    S[i] = b
                    continue

                if not a_feas and not b_feas:
                    # both infeasible -> smaller CV wins, tie-break by F
                    if cva < cvb:
                        S[i] = a
                    elif cvb < cva:
                        S[i] = b
                    else:
                        S[i] = a if Fa[0] <= Fb[0] else b
                    continue

            # Otherwise (no CV), compare by objective only
            S[i] = a if Fa[0] <= Fb[0] else b
            continue

        # --- Case B: 3-objective layout (loss, alpha_violation, beta_violation)
        if Fa.size >= 3 and Fb.size >= 3:
            a_feas = (abs(Fa[1]) <= eps_cv) and (abs(Fa[2]) <= eps_cv)
            b_feas = (abs(Fb[1]) <= eps_cv) and (abs(Fb[2]) <= eps_cv)

            if a_feas and not b_feas:
                S[i] = a
                continue
            if b_feas and not a_feas:
                S[i] = b
                continue

            if a_feas and b_feas:
                S[i] = a if Fa[0] <= Fb[0] else b
                continue

            cva = agg_cv_from_two(Fa)
            cvb = agg_cv_from_two(Fb)
            if cva < cvb:
                S[i] = a
            elif cvb < cva:
                S[i] = b
            else:
                S[i] = a if Fa[0] <= Fb[0] else b
            continue

        # --- Mixed/unexpected shapes: fall back to compare by first objective
        S[i] = a if Fa[0] <= Fb[0] else b

    return S

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

    try:
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
            # 4) Set up the algorithm and termination criteria
            # for single-objective optimization
            pop_size = choose_de_pop_size(problem)

            selection = TournamentSelection(
                pressure=2,
                func_comp=lambda pop, P, **kw: binary_tournament_loss_cv(
                    pop, P, eps_cv=1e-10, cv_mode="linf", **kw
                )
            )

            algorithm = GA(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                selection=selection,
                crossover=SBX(prob=0.9, eta=25),
                mutation=PM(eta=40),
                eliminate_duplicates=True
            )

            # algorithm = DE(
            #     pop_size=pop_size,
            #     sampling=FloatRandomSampling(),
            #     variant="DE/rand/1/bin",
            #     CR=0.9,  # crossover rate (binomial crossover)
            #     F=0.8,  # differential weight
            #     dither="vector",  # recommended for robustness
            #     jitter=False
            # )

            termination = get_termination("n_gen", 10000)

        else:
            # Multi-objective optimization (UNSGA3)
            pop_size = choose_nsga_pop_size(problem)

            selection = TournamentSelection(
                pressure=2,
                func_comp=lambda pop, P, **kw: binary_tournament_loss_cv(
                    pop, P, eps_cv=1e-10, cv_mode="linf", **kw
                )
            )

            algorithm = UNSGA3(
                ref_dirs=get_reference_directions("das-dennis", problem.n_obj, n_partitions=12),
                pop_size=pop_size,
                sampling=LHS(),
                selection=selection,
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True,
            )

            termination = get_termination("n_gen", 2000)

        # 5) Run the optimization
        result = _run_with_ctrlc(problem, algorithm, termination, verbose=True, save_history=True)

        # 6) Grab execution time and close the pool
        # Convert execution time to minutes and hours
        exec_time_seconds = result.exec_time
        exec_time_minutes = exec_time_seconds / 60
        exec_time_hours = exec_time_seconds / 3600
        logger.info(f"Execution Time: {exec_time_seconds:.2f} seconds |  "
                    f"{exec_time_minutes:.2f} minutes |  "
                    f"{exec_time_hours:.2f} hours")
    finally:
        pool.close()
        pool.join()

    return problem, result

def pick_best_loss_with_constraints_as_objectives(
    result,
    eps_cv=1e-10,
    cv_mode="l1",              # "l1" | "linf" | "l2"
    tie_tol=1e-12,
    tie_break="loss_then_l2",  # "loss_then_l2" | "loss_only"
):
    """
    Select best solution when:
      F[:,0] = loss (minimize)
      F[:,1] = constraint violation 1 (minimize, ideally 0)
      F[:,2] = constraint violation 2 (minimize, ideally 0)

    Rule:
      A) If any feasible (cv1<=eps and cv2<=eps): choose min loss among feasible.
      B) Else: choose min aggregated CV; tie-break by loss; optional tie-break by ||X||2.

    Returns:
      best_solution, best_index_in_pop, info
    """
    pop = result.pop
    F = np.asarray([ind.F for ind in pop], dtype=float)
    X = np.asarray([ind.X for ind in pop], dtype=float)

    if F.shape[1] < 3:
        raise ValueError(f"Expected at least 3 objectives (loss, cv1, cv2). Got {F.shape[1]}")

    loss = F[:, 0]
    cv1 = np.abs(F[:, 1])
    cv2 = np.abs(F[:, 2])

    feas = (cv1 <= eps_cv) & (cv2 <= eps_cv)

    info = {
        "n_total": len(pop),
        "n_feasible": int(np.sum(feas)),
        "eps_cv": float(eps_cv),
    }

    # --- A) Feasible exists: minimize loss
    if np.any(feas):
        idx = np.where(feas)[0]
        best_local = idx[np.argmin(loss[idx])]
        best_solution = pop[int(best_local)]

        info.update({
            "selection_case": "feasible_min_loss",
            "best_index": int(best_local),
            "best_F": np.asarray(best_solution.F, dtype=float),
        })
        return best_solution, int(best_local), info

    # --- B) No feasible: minimize aggregated violation, tie-break by loss
    if cv_mode == "l1":
        agg = cv1 + cv2
    elif cv_mode == "linf":
        agg = np.maximum(cv1, cv2)
    elif cv_mode == "l2":
        agg = np.sqrt(cv1**2 + cv2**2)
    else:
        raise ValueError("cv_mode must be one of: 'l1', 'linf', 'l2'")

    best_agg = float(np.min(agg))
    cand = np.where(np.abs(agg - best_agg) <= tie_tol)[0]

    if len(cand) > 1:
        # tie-break by loss
        cand = cand[np.argsort(loss[cand])]
        if tie_break == "loss_then_l2" and len(cand) > 1:
            # if still tied, prefer smaller parameter norm
            l2 = np.linalg.norm(X[cand], axis=1)
            cand = cand[np.argsort(l2)]

    best_idx = int(cand[0])
    best_solution = pop[best_idx]

    info.update({
        "selection_case": "infeasible_min_violation_then_loss",
        "cv_mode": cv_mode,
        "best_index": best_idx,
        "best_F": np.asarray(best_solution.F, dtype=float),
        "best_agg_cv": best_agg,
    })
    return best_solution, best_idx, info

def post_optimization_nsga(
        result,
        weights=np.array([1.0, 1.0, 1.0]),
        ref_point=np.array([3, 1, 1])):
    """
    Post-processes the result of a multi-objective optimization run.

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
    best_solution, best_index, sel_info = pick_best_loss_with_constraints_as_objectives(
        result,
        eps_cv=1e-10,
        cv_mode="linf",  # recommended: minimizes the worst constraint
        tie_break="loss_then_l2"
    )

    logger.info(f"Best solution selected by: {sel_info['selection_case']} with index: {best_index}")
    logger.info(f"Best solution F: {sel_info['best_F']}")

    scores = sel_info["best_F"]
    best_objectives = np.asarray(best_solution.F, dtype=float)
    optimized_params = best_solution.X

    # Additional references from the result
    F = result.F  # The entire final objective set
    pairs = [(0, 1), (0, 2), (1, 2)]
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    # Decomposition objects
    decomp = ASF()
    asf_i = decomp.do(F, 1 / weights).argmin()

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
    for i, individual in enumerate(result.pop):  # Displaying first 5 individuals for brevity
        row = {"Individual": i + 1, "Objective Value (F)": float(individual.F[0])}  # Add individual info
        row.update({f"α_{j}": float(x) for j, x in enumerate(individual.X)})  # Add decision variables
        pops.append(row)

    waterfall_df = pd.DataFrame(pops)
    waterfall_df.to_csv(f"{OUT_DIR}/parameter_scan.csv", index=False)

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


def post_optimization_de(
        result,
        alpha_values,
        beta_values):
    """
    Post-processes the result of a multi-objective optimization run.

    Args:
        result: The final result object from the optimizer (e.g., a pymoo result).

    Returns:
        dict: A dictionary with keys:
            'best_solution': The best individual from the weighted scoring.
            'best_objectives': Its corresponding objective vector.
            'optimized_params': The individual's decision variables (X).
            'scores': Weighted scores for each solution in the Pareto front.
            'best_index': The index of the best solution according to weighted score.
            'hist_hv': The hyper volume per generation.
            'hist_igd': The IGD+ per generation.
            'convergence_df': The DataFrame with iteration vs. best objective value
                for each iteration in the result history.
    """
    # # Display key results
    # print("Optimization Results:\n")
    # print("Objective Value (F):", result.F)  # Best objective value
    # print("Best Solution Variables (X):", result.X)  # Best solution variables
    # print("Constraint Violations (G):", result.G)  # Constraint violations
    # print("Constraint Violation Summary (CV):", result.CV)  # Summary of constraint violations
    # # Additional attributes
    # print("\nAdditional Information:")
    # print("Algorithm:", result.algorithm)
    # print("Archive:", result.archive)
    # print("Feasibility (feas):", result.feas)
    # # Display details about the population with more attributes for each individual
    # print("\nFirst 5 Population Details:")
    # for i, individual in enumerate(result.pop[:5]):  # Displaying first 5 individuals for brevity
    #     print(f"\nIndividual {i + 1}:")
    #     print("  Decision Variables (X):", individual.X)
    #     print("  Objective Value (F):", individual.F)
    #     print("  Constraint Violation (CV):", individual.CV if hasattr(individual, 'CV') else "N/A")
    #     print("  Feasibility (feas):", individual.feas if hasattr(individual, 'feas') else "N/A")
    # Combined alpha and beta labels with Greek symbols
    param_labels = []
    # Add alpha (α) labels
    for (gene, psite), kinases in alpha_values.items():
        for kinase in kinases.keys():
            param_labels.append(f"α_{gene}_{psite}_{kinase}")
    # Add beta (β) labels
    for (kinase, psite), _ in beta_values.items():
        param_labels.append(f"β_{kinase}_{psite}")
    # Initialize an empty list to store individual data
    pops = []
    # Loop through the first 5 individuals and extract data
    for i, individual in enumerate(result.pop):  # Displaying first 5 individuals for brevity
        row = {"Individual": i + 1, "Objective Value (F)": individual.F[0]}  # Add individual info
        row.update({param_labels[j]: x for j, x in enumerate(individual.X)})  # Add decision variables
        pops.append(row)
    # Create a DataFrame from the collected data
    waterfall_df = pd.DataFrame(pops)
    waterfall_df.to_csv(f'{OUT_DIR}/parameter_scan.csv', index=False)
    # Visualize the convergence
    val = [e.opt.get("F")[0] for e in result.history]
    # Flatten the list or extract the first element of each value
    flattened_val = [v[0] if isinstance(v, (list, np.ndarray)) else v for v in val]
    # Correctly construct the DataFrame
    convergence_df = pd.DataFrame({
        "Iteration": np.arange(len(flattened_val)),
        "Value": flattened_val
    })
    # Save the DataFrame to a CSV file
    convergence_df.to_csv(f'{OUT_DIR}/convergence.csv', index=False)
    ordered_optimizer_runs = waterfall_df.sort_values(by="Objective Value (F)", ascending=True)
    objective_values = ordered_optimizer_runs["Objective Value (F)"].values
    # Dynamically determine the threshold based on the range of objective values
    threshold = 0.05 * (objective_values.max() - objective_values.min())
    # Determine indices to plot
    indices_to_plot = [0]  # Always plot the first point
    for i in range(1, len(objective_values)):
        if abs(objective_values[i] - objective_values[i - 1]) > threshold:  # Significant change
            indices_to_plot.append(i)
        elif i % 10 == 0:  # Plot sparsely for small changes
            indices_to_plot.append(i)
    # Plot only the selected indices
    x_values = indices_to_plot
    y_values = [objective_values[i] for i in indices_to_plot]
    # Melt the DataFrame to make it long-form for easy plotting
    long_df = waterfall_df.melt(id_vars=["Individual", "Objective Value (F)"],
                                value_vars=param_labels,
                                var_name="Parameter",
                                value_name="Parameter Value")
    # Add a column to classify parameters as 'α' or 'β'
    long_df["Type"] = long_df["Parameter"].apply(
        lambda x: "α" if x.startswith("α") else ("β" if x.startswith("β") else "Other"))
    # Sort the DataFrame by "Parameter Value"
    long_df = long_df.sort_values(by="Objective Value (F)")

    return (
        ordered_optimizer_runs,
        convergence_df,
        long_df,
        x_values,
        y_values,
        val
    )
