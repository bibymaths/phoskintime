from scipy.optimize import minimize

def run_optimization(obj_fun, params_initial, opt_method, bounds, constraints):
    result = minimize(obj_fun, params_initial, method=opt_method,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 20000, 'verbose': 3} if opt_method == "trust-constr" else {'maxiter': 20000})
    return result, result.x