from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions

from tfopt.evol.config.logconf import setup_logger

logger = setup_logger()

def _choose_partitions(n_obj: int) -> int:
    """
    Heuristic for Das-Dennis partitions.
    Larger => more ref directions => larger required population.
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
    Run the optimization using the specified algorithm and problem.

    Args:
        problem (Problem): The optimization problem to solve.
        total_dim (int): Total number of dimensions in the problem.
        optimizer (int): The optimizer to use (0 for NSGA2, 1 for SMSEMOA, 2 for AGEMOEA).

    Returns:
        res (Result): The result of the optimization.
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

    termination = DefaultMultiObjectiveTermination()

    # Run the optimization
    res = pymoo_minimize(problem=problem,
                         algorithm=algo,
                         termination=termination,
                         seed=1,
                         verbose=True)

    return res
