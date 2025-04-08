from kinopt.evol.config import METHOD

if METHOD == "DE":
    from kinopt.evol.objfn.minfndiffevo import _estimated_series, _residuals
else:
    from kinopt.evol.objfn.minfnnsgaii import _estimated_series, _residuals

estimated_series = _estimated_series
residuals = _residuals
