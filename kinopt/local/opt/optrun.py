from scipy.optimize import minimize


def run_optimization(obj_fun, params_initial, opt_method, bounds, constraints):
    """
    Run optimization using the specified method.

    Args:
        obj_fun: Objective function to minimize.
        params_initial: Initial parameters for the optimization.
        opt_method: Optimization method to use (e.g., 'SLSQP', 'trust-constr').
        bounds: Bounds for the parameters.
        constraints: Constraints for the optimization.

    Returns:
        result: Result of the optimization.
        optimized_params: Optimized parameters.
    """
    result = minimize(obj_fun, params_initial, method=opt_method,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 20000, 'verbose': 3} if opt_method == "trust-constr" else {'maxiter': 20000})
    return result, result.x
