from scipy.optimize import minimize, differential_evolution
from tfopt.local.objfn.minfn import objective_wrapper


def run_optimizer(x0, bounds, lin_cons, expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type):
    """
    Runs the optimization algorithm to minimize the objective function.

    Parameters:
      x0                  : Initial guess for the optimization variables.
      bounds              : Bounds for the optimization variables.
      lin_cons            : Linear constraints for the optimization problem.
      expression_matrix   : (n_genes x T_use) measured gene expression values.
      regulators          : (n_genes x n_reg) indices of TF regulators for each gene.
      tf_protein_matrix   : (n_TF x T_use) TF protein time series.
      psite_tensor        : (n_TF x n_psite_max x T_use) matrix of PSite signals (padded with zeros).
      n_reg               : Maximum number of regulators per gene.
      T_use               : Number of time points used.
      n_genes, n_TF     : Number of genes and TF respectively.
      beta_start_indices  : Integer array giving the starting index (in the β–segment) for each TF.
      num_psites          : Integer array with the actual number of PSites for each TF.
      loss_type           : Type of loss function to use.
    Returns:
        result             : Result of the optimization process, including the optimized parameters and objective value.
    """
    m = "SLSQP" # or trust-constr or SLSQP
    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        args=(expression_matrix, regulators, tf_protein_matrix, psite_tensor, n_reg, T_use, n_genes, beta_start_indices, num_psites, loss_type),
        method=m,
        bounds=bounds,
        constraints=lin_cons,
        options={"disp": True, "maxiter": 10000} if m == "SLSQP" else {"disp": True, "maxiter": 10000, "verbose": 3}
    )
    return result