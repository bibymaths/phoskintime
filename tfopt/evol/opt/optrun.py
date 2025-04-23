import contextlib
import io

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.algorithms.moo.nsga2 import NSGA2
from tfopt.evol.config.logconf import setup_logger

logger = setup_logger()


def run_optimization(problem, total_dim, optimizer):
    """
    Run the optimization using the specified algorithm and problem.
    This function sets up the algorithm parameters, initializes the optimizer,
    and runs the optimization process.

    :param problem:
    :param total_dim:
    :param optimizer:
    :return: result
    """
    # Define algorithm settings.
    global algo
    pop_size = total_dim*2
    crossover = TwoPointCrossover(prob=0.9)
    mutation = PolynomialMutation(prob=1.0 / total_dim, eta=20)
    eliminate_duplicates = True

    # Choose the optimizer based on the input parameter.
    if optimizer == 0:
        # NSGA2 settings.
        algo = NSGA2(
            pop_size=pop_size,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates
        )
    elif optimizer == 1:
        # SMSEMOA settings.
        algo= SMSEMOA(
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
    # buf = io.StringIO()
    # with contextlib.redirect_stdout(buf):
    res = pymoo_minimize(problem=problem,
                         algorithm=algo,
                         termination=termination,
                         seed=1,
                         verbose=True)

        # Log the captured pymoo progress
    # pymoo_progress = buf.getvalue()
    # if pymoo_progress.strip():  # only log if there's actual text
    #     logger.info("--- Progress Output ---\n" + pymoo_progress)

    return res