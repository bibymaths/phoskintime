from abopt.evol.config import METHOD

if METHOD == "DE":
    from abopt.evol.objfn.minfndiffevo import _estimated_series, _residuals
else:
    from abopt.evol.objfn.minfnnsgaii import _estimated_series, _residuals

estimated_series = _estimated_series
residuals = _residuals
