import numpy as np
import multiprocessing as mp
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population
from pymoo.termination.default import DefaultMultiObjectiveTermination


def get_refined_bounds(X, current_xl, current_xu, padding=0.2):
    """
    Calculates tighter bounds around the existing Pareto set (X).
    """
    p_min = np.min(X, axis=0)
    p_max = np.max(X, axis=0)

    span = p_max - p_min
    span[span < 1e-6] = 0.1

    new_xl = p_min - (span * padding)
    new_xu = p_max + (span * padding)

    new_xl = np.maximum(new_xl, current_xl)
    new_xu = np.minimum(new_xu, current_xu)

    vol_old = np.prod(current_xu - current_xl + 1e-9)
    vol_new = np.prod(new_xu - new_xl + 1e-9)
    ratio = vol_old / vol_new
    print(f"[Refine] Search space volume reduced by factor: {ratio:.2e}")

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


def run_refinement(problem, prev_res, args, padding=0.25):
    """
    Main driver with Parallel Processing support.
    """
    print("\n" + "=" * 40)
    print("       STARTING REFINEMENT (ZOOM-IN)      ")
    print("=" * 40)

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
        print(f"[Refine] Parallel evaluation enabled with {args.cores} workers.")
    else:
        print("[Refine] Parallel evaluation disabled.")

    try:
        # 3. Generate Hybrid Population
        print(f"[Refine] Creating hybrid population (Size: {args.pop})")
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
        print(f"[Refine] Running for {args.n_gen} generations...")

        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=0.0025,
            period=30,
            n_max_gen=args.n_gen,
            n_max_evals=100000
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
