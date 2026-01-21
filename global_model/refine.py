import numpy as np
import multiprocessing as mp
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population
from pymoo.termination.default import DefaultMultiObjectiveTermination

from config.config import setup_logger
from global_model.config import RESULTS_DIR

logger = setup_logger(log_dir=RESULTS_DIR)


def get_refined_bounds(X, current_xl, current_xu, idx, padding=0.2):
    """
    Refines bounds and prints a tree-like biological report of the network parameters.
    """
    if hasattr(X, "values"):
        X = X.values

    # Calculate log-space bounds
    p_min, p_max = np.min(X, axis=0), np.max(X, axis=0)
    span = np.maximum(p_max - p_min, 1e-2)
    new_xl = np.maximum(p_min - (span * padding), current_xl)
    new_xu = np.minimum(p_max + (span * padding), current_xu)

    logger.info("=" * 90)
    logger.info("   NETWORK TOPOLOGY REFINEMENT REPORT (Physical Rates)")
    logger.info("=" * 90)

    # Pointer to track our position in the 1094+ parameter vector
    ptr = 0

    # 1. UPSTREAM: Kinase Inputs
    logger.info("[STAGE 1: KINASE MULTIPLIERS (c_k)]")
    logger.info(f"{'Entity (Kinase)':<30} {'Min Multiplier':<15} {'Max Multiplier':<15}")
    for k_name in idx.kinases:
        lo, hi = np.exp(new_xl[ptr]), np.exp(new_xu[ptr])
        logger.info(f"  {k_name:<30} {lo:<15.2e} {hi:<15.2e}")
        ptr += 1

    # 2. NODES: Protein & Phosphosite Turnover
    logger.info("[STAGE 2: PROTEIN & PHOSPHO KINETICS]")
    for i, p_name in enumerate(idx.proteins):
        logger.info(f"==== > Node: {p_name} (Protein/TF)")

        # Core Protein Params (A, B, C, D)
        a_lo, a_hi = np.exp(new_xl[ptr]), np.exp(new_xu[ptr])  # Synthesis
        b_lo, b_hi = np.exp(new_xl[ptr + 1]), np.exp(new_xu[ptr + 1])  # Degradation
        c_lo, c_hi = np.exp(new_xl[ptr + 2]), np.exp(new_xu[ptr + 2])  # Phosphorylation
        d_lo, d_hi = np.exp(new_xl[ptr + 3]), np.exp(new_xu[ptr + 3])  # Dephosphorylation

        logger.info(f"  - Production (A_i)  : [{a_lo:.2e} to {a_hi:.2e}]")
        logger.info(f"  - Decay (B_i)       : [{b_lo:.2e} to {b_hi:.2e}]")
        logger.info(f"  - Phos Sensitivity  : [{c_lo:.2e} to {c_hi:.2e}]")
        logger.info(f"  - Dephos Baseline   : [{d_lo:.2e} to {d_hi:.2e}]")
        ptr += 4

        # Site-specific parameters (Dp_i)
        n_sites = idx.n_sites[i]
        if n_sites > 0:
            logger.info(f"  - Phosphosites ({n_sites}):")
            for s_idx in range(n_sites):
                site_label = idx.sites[i][s_idx]
                dp_lo, dp_hi = np.exp(new_xl[ptr]), np.exp(new_xu[ptr])
                logger.info(f"    * {site_label:<20} Dp_i (Site Decay): [{dp_lo:.2e} to {dp_hi:.2e}]")
                ptr += 1

        # Regulatory Effect (E_i)
        e_lo, e_hi = np.exp(new_xl[ptr]), np.exp(new_xu[ptr])
        logger.info(f"  - TF Regulatory Power (E_i): [{e_lo:.2e} to {e_hi:.2e}]")
        ptr += 1

    # 3. GLOBAL: TF Scale
    logger.info("[STAGE 3: GLOBAL SYSTEM SCALING]")
    tf_lo, tf_hi = np.exp(new_xl[ptr]), np.exp(new_xu[ptr])
    logger.info(f"  Global TF Scale Factor: [{tf_lo:.4f} to {tf_hi:.4f}]")

    return new_xl, new_xu


def create_multistart_population(X_best, pop_size, new_xl, new_xu):
    """
    Creates hybrid population: 50% Warm Start, 50% Fresh Sampling.
    """
    n_best = len(X_best)
    n_warm = int(pop_size * 0.5)
    n_fresh = pop_size - n_warm

    # Warm Start
    if n_best >= n_warm:
        indices = np.random.choice(n_best, n_warm, replace=False)
        X_warm = X_best[indices]
    else:
        X_warm = np.zeros((n_warm, X_best.shape[1]))
        X_warm[:n_best] = X_best
        n_needed = n_warm - n_best
        src_indices = np.random.randint(0, n_best, n_needed)
        noise = np.random.normal(0, 0.05, (n_needed, X_best.shape[1])) * (new_xu - new_xl)
        X_warm[n_best:] = X_best[src_indices] + noise

    X_warm = np.clip(X_warm, new_xl, new_xu)

    # Fresh Sampling
    val = np.random.random((n_fresh, len(new_xl)))
    X_fresh = new_xl + val * (new_xu - new_xl)

    X_pop = np.vstack([X_warm, X_fresh])
    return Population.new("X", X_pop)


def run_iterative_refinement(problem, prev_res, args, idx = None, max_passes=3, padding=0.25):
    """
    Runs refinement recursively to zoom in on the optimum.

    Args:
        idx: Index object containing parameter metadata.
        max_passes: How many times to re-zoom (e.g., 3 times).
        padding: The boundary padding factor (decreases slightly each pass).
    """
    current_res = prev_res

    # We must explicitly attach the runner ONCE here to avoid overhead
    pool = None
    if args.cores > 1:
        pool = mp.Pool(args.cores)
        problem.elementwise_runner = StarmapParallelization(pool.starmap)
        logger.info(f"[Refine] Parallel evaluation enabled with {args.cores} workers.")

    try:
        for i in range(max_passes):
            logger.info("=" * 50)
            logger.info(f"      REFINEMENT PASS {i + 1}/{max_passes} (Zoom-in)      ")
            logger.info("=" * 50)

            # 1. Calculate New Bounds based on previous result
            # We decay the padding slightly to zoom in harder on later passes
            current_padding = max(0.05, padding * (0.8 ** i))
            new_xl, new_xu = get_refined_bounds(current_res.X, problem.xl, problem.xu, idx=idx, padding=current_padding)

            # Check if volume is effectively zero (converged)
            if np.allclose(new_xl, new_xu, atol=1e-5):
                logger.info("[Refine] Search space collapsed to a point. Stopping refinement.")
                break

            # Update Problem Bounds
            problem.xl = new_xl
            problem.xu = new_xu

            # 2. Hybrid Population (Warm Start + Exploration)
            # Pass 1: 50/50. Later passes: 80/20 (trust the warm start more)
            current_pop = create_multistart_population(current_res.X, args.pop, new_xl, new_xu)

            # 3. Setup Algorithm
            # We increase eta (stiffness) each pass to encourage fine-tuning over jumping
            current_eta = 20 + (i * 10)

            algorithm = UNSGA3(
                ref_dirs=current_res.algorithm.ref_dirs,  # Keep reference directions
                pop_size=args.pop,
                sampling=current_pop,
                crossover=SBX(prob=0.9, eta=current_eta),
                mutation=PM(prob=1 / problem.n_var, eta=current_eta + 5),
                eliminate_duplicates=True
            )

            # 4. Termination Criteria
            # We can run fewer generations in later passes as the space is smaller
            # E.g. Pass 1: 100%, Pass 2: 75%, Pass 3: 50%
            gen_scale = max(0.5, 1.0 - (i * 0.2))
            pass_gens = int(args.n_gen * gen_scale)

            termination = DefaultMultiObjectiveTermination(
                xtol=1e-8,
                cvtol=1e-6,
                ftol=0.001,  # Stricter tolerance
                period=20,
                n_max_gen=pass_gens,
                n_max_evals=10000000
            )

            # 5. Execute
            logger.info(f"[Refine] Running Pass {i + 1} for {pass_gens} generations...")
            res = pymoo_minimize(
                problem,
                algorithm,
                termination=termination,
                seed=args.seed + 1 + i,
                verbose=True,
                save_history=True  # Only save history for the final result usually
            )

            # Validation: Did we improve?
            # Check mean error of the best solution found
            old_best = np.min(current_res.F[:, 0]) if current_res.F is not None else float('inf')
            new_best = np.min(res.F[:, 0]) if res.F is not None else float('inf')

            logger.info(f"[Refine] Pass {i + 1} Best Objective: {new_best:.6f} (Prev: {old_best:.6f})")

            current_res = res

    except Exception as e:
        logger.error(f"[Refine] Crash during recursive refinement: {e}")
        raise e

    finally:
        # Cleanup Pool
        if pool is not None:
            pool.close()
            pool.join()
            problem.elementwise_runner = None

    return current_res

############################################################
########### DEPRECATED: Use run_iterative_refinement()  ####
############################################################
def run_refinement(problem, prev_res, args, padding=0.25):
    """
    Main driver with Parallel Processing support.
    """
    logger.info("=" * 40)
    logger.info("       STARTING REFINEMENT (ZOOM-IN)      ")
    logger.info("=" * 40)

    # 1. Update Bounds (Zoom In)
    new_xl, new_xu = get_refined_bounds(prev_res.X, problem.xl, problem.xu, padding=padding)
    problem.xl = new_xl
    problem.xu = new_xu

    # 2. Setup Parallel Runner
    # We must explicitly attach the runner to the problem
    pool = None
    if args.cores > 1:
        pool = mp.Pool(args.cores)
        problem.elementwise_runner = StarmapParallelization(pool.starmap)
        logger.info(f"[Refine] Parallel evaluation enabled with {args.cores} workers.")
    else:
        logger.info("[Refine] Parallel evaluation disabled.")

    try:
        # 3. Generate Hybrid Population
        logger.info(f"[Refine] Creating hybrid population (Size: {args.pop})")
        initial_pop = create_multistart_population(prev_res.X, args.pop, new_xl, new_xu)

        # 4. Setup Algorithm
        algorithm = UNSGA3(
            ref_dirs=prev_res.algorithm.ref_dirs,
            pop_size=args.pop,
            sampling=initial_pop,
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(prob=1 / problem.n_var, eta=25),
            eliminate_duplicates=True
        )

        # 5. Run Optimization
        logger.info(f"[Refine] Running for {args.n_gen} generations...")

        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=0.0025,
            period=30,
            n_max_gen=args.n_gen,
            n_max_evals=10000000
        )

        res = pymoo_minimize(
            problem,
            algorithm,
            termination=termination,
            seed=args.seed + 1,
            verbose=True,
            save_history=True
        )

    finally:
        # 6. Cleanup Pool
        if pool is not None:
            pool.close()
            pool.join()
            problem.elementwise_runner = None  # Clean up reference

    return res
