import numpy as np
from pymoo.core.problem import ElementwiseProblem
from kinopt.evol.config import include_regularization, lb, ub, loss_type
from kinopt.evol.optcon import n, P_initial_array

class PhosphorylationOptimizationProblem(ElementwiseProblem):
    """
    Custom optimization problem for phosphorylation analysis.

    Defines the constraints, bounds, and objective function for optimizing
    alpha and beta parameters across gene-psite-kinase relationships.

    Attributes:
        P_initial (dict): Mapping of gene-psite pairs to kinase relationships and time-series data.
        P_initial_array (np.ndarray): Observed time-series data for gene-psite pairs.
        K_index (dict): Mapping of kinases to their respective psite data.
        K_array (np.ndarray): Array containing time-series data for kinase-psite combinations.
        gene_psite_counts (list): Number of kinases per gene-psite combination.
        beta_counts (dict): Mapping of kinase indices to the number of associated psites.
    """

    def __init__(self, P_initial, P_initial_array, K_index, K_array, gene_psite_counts, beta_counts, **kwargs):
        """
        Initializes the optimization problem with given data and constraints.

        Args:
            P_initial (dict): Mapping of gene-psite pairs to kinase relationships and time-series data.
            P_initial_array (np.ndarray): Observed time-series data for gene-psite pairs.
            K_index (dict): Mapping of kinases to their respective psite data.
            K_array (np.ndarray): Array containing time-series data for kinase-psite combinations.
            gene_psite_counts (list): Number of kinases per gene-psite combination.
            beta_counts (dict): Mapping of kinase indices to the number of associated psites.
        """
        self.P_initial = P_initial
        self.P_initial_array = P_initial_array
        self.K_index = K_index
        self.K_array = K_array
        self.gene_psite_counts = gene_psite_counts
        self.beta_counts = beta_counts
        self.num_alpha = sum(gene_psite_counts)
        self.num_beta = sum(beta_counts.values())

        # Define the problem with pymoo
        super().__init__(
            n_var=self.num_alpha + self.num_beta,
            n_obj=1,  # Single objective (sum of squared residuals)
            n_ieq_constr=self.num_alpha + len(beta_counts),  # Constraints for alpha and beta
            xl=np.concatenate([(0,) * self.num_alpha, (lb,) * self.num_beta]),
            xu=np.concatenate([(1,) * self.num_alpha, (ub,) * self.num_beta])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates the objective function and constraints for the given decision variables.

        Args:
            x (np.ndarray): Decision variable vector.
            out (dict): Dictionary to store objective function value and constraint values.
        """
        # Calculate the residuals for the objective function
        error = self.objective_function(x)

        # Initialize an empty list for constraints
        constraints = []

        # Constraints for alphas (sum to 1 for each gene-psite-kinase group)
        alpha_start = 0
        for count in self.gene_psite_counts:
            constraints.append(np.sum(x[alpha_start:alpha_start + count]) - 1)
            alpha_start += count

        # Constraints for betas (sum to 1 for each kinase across its psites)
        beta_start = self.num_alpha
        for kinase, psites in self.K_index.items():
            num_psites = len(psites)
            constraints.append(np.sum(x[beta_start:beta_start + num_psites]) - 1)
            beta_start += num_psites

        # Ensure constraints list matches the expected length (self.n_ieq_constr)
        # Pad constraints with zeros if fewer constraints are defined
        constraints = np.array(constraints)
        if constraints.shape[0] < self.n_ieq_constr:
            constraints = np.concatenate([constraints, np.zeros(self.n_ieq_constr - constraints.shape[0])])

        # Output the objective and constraints, reshaping G to have shape (1, n_ieq_constr)
        out["F"] = error
        out["G"] = constraints.reshape(1, self.n_ieq_constr)

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