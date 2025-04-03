from config.constants import ODE_MODEL

if ODE_MODEL == 'distmod':
    from .initdist import initial_condition as initial_condition_impl
elif ODE_MODEL == 'succmod':
    from .initsucc import initial_condition as initial_condition_impl
elif ODE_MODEL == 'randmod':
    from .initrand import initial_condition as initial_condition_impl
elif ODE_MODEL == 'testmod':
    from .inittest import initial_condition as initial_condition_impl
else:
    raise ValueError(f"Unsupported ODE_MODEL: {ODE_MODEL}")

initial_condition = initial_condition_impl