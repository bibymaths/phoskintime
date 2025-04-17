from kinopt.evol.config import METHOD

# This module imports the appropriate functions for estimating series and calculating residuals
# based on the selected optimization method.
# The functions are imported from different modules depending on the value of METHOD.
# The available methods are "DE" (Differential Evolution) and "NSGAII" (Non-dominated Sorting Genetic Algorithm II).
if METHOD == "DE":
    from kinopt.evol.objfn.minfndiffevo import _estimated_series, _residuals
else:
    from kinopt.evol.objfn.minfnnsgaii import _estimated_series, _residuals

estimated_series = _estimated_series
residuals = _residuals
