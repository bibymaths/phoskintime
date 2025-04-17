from scipy.optimize import minimize

def run_optimization(obj_fun, params_initial, opt_method, bounds, constraints):
    """
    Run optimization using the specified method.

    :param obj_fun:
    :param params_initial:
    :param opt_method:
    :param bounds:
    :param constraints:
    :return: result, optimized parameters
    """
    result = minimize(obj_fun, params_initial, method=opt_method,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 20000, 'verbose': 3} if opt_method == "trust-constr" else {'maxiter': 20000})
    return result, result.x