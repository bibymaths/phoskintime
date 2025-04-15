from scipy.optimize import minimize, differential_evolution
from tfopt.local.objfn.minfn import objective_wrapper


def run_optimizer(x0, bounds, lin_cons, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type):

    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        args=(expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type),
        method="trust-constr",
        bounds=bounds,
        constraints=lin_cons,
        options={"disp": True, "maxiter": 10000, "verbose": 3}
    )

    return result