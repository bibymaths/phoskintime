from config.constants import ODE_MODEL, _normalize_model_name

_model = _normalize_model_name(ODE_MODEL)

if _model == 'protwise':
    # The local "protwise" model uses the distributive steady-state initializer
    # by default so ODE_MODEL="protwise" works in process_gene/runner paths.
    from .initdist import initial_condition as initial_condition_impl
elif _model == 'distmod':
    from .initdist import initial_condition as initial_condition_impl
elif _model == 'succmod':
    from .initsucc import initial_condition as initial_condition_impl
elif _model == 'randmod':
    from .initrand import initial_condition as initial_condition_impl
elif _model == 'testmod':
    from .inittest import initial_condition as initial_condition_impl
else:
    raise ValueError(f"Unsupported ODE_MODEL: {ODE_MODEL}")

initial_condition = initial_condition_impl
