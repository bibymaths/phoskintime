from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from tfopt.evol.config.logconf import setup_logger

logger = setup_logger()

def _choose_partitions(n_obj: int) -> int:
    """
    Determine the number of partitions for Das-Dennis reference directions.

    This heuristic selects an appropriate number of partitions based on the
    number of objectives in the optimization problem. Larger partition values
    generate more reference directions, which in turn require larger population
    sizes for the algorithm to function effectively.

    Args:
        n_obj (int): Number of objectives in the multi-objective optimization problem.

    Returns:
        int: Number of partitions to use for Das-Dennis reference direction generation.
            Returns 50 for 1-2 objectives, 12 for 3 objectives, 8 for 4 objectives,
            6 for 5 objectives, and 4 for 6 or more objectives.
    """
    if n_obj <= 2:
        return 50
    if n_obj == 3:
        return 12
    if n_obj == 4:
        return 8
    if n_obj == 5:
        return 6
    return 4

def run_optimization(problem, total_dim, optimizer):
    """
    Execute multi-objective optimization using the specified algorithm.

    This function configures and runs one of three multi-objective evolutionary
    algorithms (UNSGA3, SMSEMOA, or AGEMOEA) on the provided optimization problem.
    The algorithm is configured with appropriate genetic operators (two-point
    crossover and polynomial mutation) and terminated after 1000 generations.

    Args:
        problem (Problem): The pymoo Problem instance defining the optimization
            problem, including objectives, constraints, and variable bounds.
        total_dim (int): Total number of decision variables (dimensions) in the
            optimization problem. Used to determine population size and mutation
            probability.
        optimizer (int): Selector for the optimization algorithm:
            - 0: UNSGA3 (Unified NSGA-III) - Reference direction-based algorithm
            - 1: SMSEMOA - S-Metric Selection Evolutionary Multi-objective Algorithm
            - 2: AGEMOEA - Adaptive Geometry Estimation-based Multi-objective
                          Evolutionary Algorithm

    Returns:
        Result: A pymoo Result object containing the optimization outcomes, including:
            - X: Decision variables of the Pareto-optimal solutions
            - F: Objective function values of the Pareto-optimal solutions
            - algorithm: The algorithm instance used
            - Additional statistics and convergence information

    Notes:
        - Population size is set to 2 * total_dim (or larger for UNSGA3 if needed)
        - Crossover probability: 0.9
        - Mutation probability: 1.0 / total_dim
        - Mutation distribution index (eta): 20
        - Termination: Fixed at 1000 generations
        - Random seed: 1 (for reproducibility)
        - Duplicate elimination is enabled for all algorithms
        - UNSGA3 automatically adjusts population size to match reference directions
    """
    # Define algorithm settings.
    global algo
    pop_size = total_dim * 2
    crossover = TwoPointCrossover(prob=0.9)
    mutation = PolynomialMutation(prob=1.0 / total_dim, eta=20)
    eliminate_duplicates = True

    # Choose the optimizer based on the input parameter.
    if optimizer == 0:
        # UNSGA3 settings.
        n_obj = int(problem.n_obj)

        n_partitions = _choose_partitions(n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)

        # UNSGA3 requires pop_size >= number of reference directions
        pop_size = max(pop_size, len(ref_dirs))

        algo = UNSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates
        )
    elif optimizer == 1:
        # SMSEMOA settings.
        algo = SMSEMOA(
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates
        )
    elif optimizer == 2:
        # AGEMOEA settings.
        algo = AGEMOEA(
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates
        )
    else:
        logger.error("Unknown optimizer type. Please choose 0 (NSGA2), 1 (SMSEMOA), or 2 (AGEMOEA).")

    termination = get_termination("n_gen", 1000)

    # Run the optimization
    res = pymoo_minimize(problem=problem,
                         algorithm=algo,
                         termination=termination,
                         seed=1,
                         verbose=True)

    return res
