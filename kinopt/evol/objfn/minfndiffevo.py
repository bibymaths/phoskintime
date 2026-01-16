import numpy as np
from pymoo.core.problem import ElementwiseProblem
from kinopt.evol.config import include_regularization, lb, ub, loss_type
from kinopt.evol.optcon import n, P_initial_array

class PhosphorylationOptimizationProblem(ElementwiseProblem):
    """
    Single-objective constrained optimization.

    Objective:
      - minimize loss

    Constraints (inequalities g(x) <= 0):
      - for each alpha group: |sum(alpha_group) - 1| <= eps
      - for each kinase beta group: |sum(beta_group) - 1| <= eps
    """

    def __init__(self, P_initial, P_initial_array, K_index, K_array,
                 gene_psite_counts, beta_counts, eps_eq=1e-10, **kwargs):

        self.P_initial = P_initial
        self.P_initial_array = P_initial_array
        self.K_index = K_index
        self.K_array = K_array
        self.gene_psite_counts = gene_psite_counts
        self.beta_counts = beta_counts

        self.num_alpha = int(sum(gene_psite_counts))
        self.num_beta = int(sum(beta_counts.values()))
        self.eps_eq = float(eps_eq)

        # number of equality groups
        n_alpha_groups = len(gene_psite_counts)
        n_beta_groups = len(K_index)  # one per kinase in K_index

        # each equality -> 2 inequalities
        n_ieq = 2 * (n_alpha_groups + n_beta_groups)

        xl = np.concatenate([np.zeros(self.num_alpha), np.full(self.num_beta, lb)], axis=0)
        xu = np.concatenate([np.ones(self.num_alpha),  np.full(self.num_beta, ub)], axis=0)

        super().__init__(
            n_var=self.num_alpha + self.num_beta,
            n_obj=1,
            n_ieq_constr=n_ieq,
            xl=xl,
            xu=xu,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        loss = self.objective_function(x)
        out["F"] = [loss]

        # Build constraints g(x) <= 0
        g = []

        # alpha group equalities
        alpha_start = 0
        for count in self.gene_psite_counts:
            s = float(np.sum(x[alpha_start:alpha_start + count]))
            # s ~= 1  ->  s - 1 <= eps  and  1 - s <= eps
            g.append((s - 1.0) - self.eps_eq)
            g.append((1.0 - s) - self.eps_eq)
            alpha_start += count

        # beta group equalities
        beta_start = self.num_alpha
        for kinase, psites in self.K_index.items():
            m = len(psites)
            s = float(np.sum(x[beta_start:beta_start + m]))
            g.append((s - 1.0) - self.eps_eq)
            g.append((1.0 - s) - self.eps_eq)
            beta_start += m

        out["G"] = np.asarray(g, dtype=float)

    def objective_function(self, params):
        """
        Computes the loss value for the given parameters using the selected loss type.

        Args:
            params (np.ndarray): Decision variables vector.

        Returns:
            float: Computed loss value.
        """
        alpha, beta = {}, {}
        alpha_start, beta_start = 0, self.num_alpha

        # Extract alphas for each gene-psite-kinase combination
        alpha = []
        for count in self.gene_psite_counts:
            alpha.append(params[alpha_start:alpha_start + count])
            alpha_start += count

        # Extract betas for each kinase-psite combination
        for idx, count in self.beta_counts.items():
            beta[idx] = params[beta_start:beta_start + count]
            beta_start += count

        # Calculate predicted matrix using alpha and beta values
        i_max, t_max = self.P_initial_array.shape
        P_i_t_matrix = np.zeros((i_max, t_max))

        for i, ((gene, psite), data) in enumerate(self.P_initial.items()):
            kinases = data['Kinases']
            gene_psite_prediction = np.zeros(t_max, dtype=np.float64)

            # Sum contributions of each kinase for the gene-psite
            for j, kinase in enumerate(kinases):
                kinase_psites = self.K_index.get(kinase)
                if kinase_psites is None:
                    continue

                # Sum contributions across all psites of the kinase
                for k_idx, (k_psite, k_time_series) in enumerate(kinase_psites):
                    kinase_betas = beta[k_idx]
                    gene_psite_prediction += alpha[i][j] * kinase_betas * k_time_series

            P_i_t_matrix[i, :] = gene_psite_prediction

        # Clip negative values to zero
        np.clip(P_i_t_matrix, a_min=0, a_max=None, out=P_i_t_matrix)

        # Calculate residuals and sum of squared errors
        residuals = self.P_initial_array - P_i_t_matrix

        # Select the loss function based on global loss_type
        if loss_type == "base":
            # MSE
            return np.sum((residuals) ** 2) / n
        elif loss_type == "base" and include_regularization:
            # MSE + L1 penalty (absolute values) + L2 penalty (squared values)
            return np.sum((residuals) ** 2) / n + np.sum(np.abs(params)) + np.sum((params) ** 2)
        elif loss_type == "autocorrelation":
            # Autocorrelation Loss
            return np.sum([np.corrcoef(residuals[i, :-1], residuals[i, 1:])[0, 1] ** 2 for i in range(i_max)])
        elif loss_type == "autocorrelation" and include_regularization:
            # Autocorrelation Loss + L1 penalty (absolute values) + L2 penalty (squared values)
            return np.sum([np.corrcoef(residuals[i, :-1], residuals[i, 1:])[0, 1] ** 2 for i in range(i_max)]) + np.sum(
                np.abs(params)) + np.sum((params) ** 2)
        elif loss_type == "huber":
            # Huber Loss
            return np.mean(np.where(
                np.abs(residuals) <= 1.0,  # Delta (adjust as necessary)
                0.5 * residuals ** 2,
                1.0 * (np.abs(residuals) - 0.5 * 1.0)
            ))
        elif loss_type == "huber" and include_regularization:
            # Huber Loss + L1 penalty (absolute values) + L2 penalty (squared values)
            return np.mean(np.where(
                np.abs(residuals) <= 1.0,  # Delta (adjust as necessary)
                0.5 * residuals ** 2,
                1.0 * (np.abs(residuals) - 0.5 * 1.0)
            )) + np.sum(np.abs(params)) + np.sum((params) ** 2)
        elif loss_type == "mape":
            # MAPE
            return np.mean(np.abs(residuals / (self.P_initial_array + 1e-12))) * 100
        elif loss_type == "mape" and include_regularization:
            # MAPE + L1 penalty (absolute values) + L2 penalty (squared values)
            return np.mean(np.abs(residuals / (self.P_initial_array + 1e-12))) * 100 + np.sum(np.abs(params)) + np.sum(
                (params) ** 2)


# Function to calculate the estimated series using optimized alpha and beta values
def _estimated_series(params, P_initial, K_index, K_array, gene_psite_counts, beta_counts):
    """
    Calculates the estimated time series for each gene-psite based on the optimized parameters.

    Args:
        params (np.ndarray): Optimized parameter vector containing alphas and betas.
        P_initial (dict): Dictionary with keys as (gene, psite) and values containing 'Kinases' and 'TimeSeries'.
        K_index (dict): Dictionary mapping each kinase to a list of (psite, time_series) tuples.
        K_array (np.ndarray): Array of kinase-psite time-series data.
        gene_psite_counts (list): List of integers indicating the number of kinases associated with each gene-psite.
        beta_counts (dict): Dictionary indicating how many beta values correspond to each kinase-psite combination.

    Returns:
        np.ndarray: Estimated time series matrix (i_max x t_max) for all gene-psite combinations.
    """
    alpha, beta = {}, {}
    alpha_start, beta_start = 0, sum(gene_psite_counts)

    # Extract alphas for each gene-psite-kinase combination
    alpha = []
    for count in gene_psite_counts:
        alpha.append(params[alpha_start:alpha_start + count])
        alpha_start += count

    # Extract betas for each kinase-psite combination
    for idx, count in beta_counts.items():
        beta[idx] = params[beta_start:beta_start + count]
        beta_start += count

    # Calculate estimated time series
    i_max, t_max = P_initial_array.shape
    P_i_t_estimated = np.zeros((i_max, t_max))

    for i, ((gene, psite), data) in enumerate(P_initial.items()):
        kinases = data['Kinases']
        gene_psite_prediction = np.zeros(t_max, dtype=np.float64)

        # Sum contributions of each kinase for the gene-psite
        for j, kinase in enumerate(kinases):
            kinase_psites = K_index.get(kinase)
            if kinase_psites is None:
                continue

            # Sum contributions across all psites of the kinase
            for k_idx, (k_psite, k_time_series) in enumerate(kinase_psites):
                kinase_betas = beta[k_idx]
                gene_psite_prediction += alpha[i][j] * kinase_betas * k_time_series

        P_i_t_estimated[i, :] = gene_psite_prediction

    return P_i_t_estimated


# Function to calculate residuals
def _residuals(P_initial_array, P_estimated):
    """
    Calculates the residuals (difference between observed and estimated values).

    Args:
        P_initial_array (np.ndarray): Observed gene-psite data.
        P_estimated (np.ndarray): Estimated gene-psite data from the model.

    Returns:
        np.ndarray: Residuals matrix (same shape as P_initial_array).
    """
    return P_initial_array - P_estimated
