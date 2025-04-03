
import pytest
from steady.initsucc import initial_condition

def verify_initial_condition_returns_correct_length_for_multiple_psites():
    num_psites = 3
    init_cond = initial_condition(num_psites)
    assert isinstance(init_cond, list)
    assert len(init_cond) == num_psites + 2

def verify_initial_condition_values_meet_lower_bound():
    num_psites = 3
    init_cond = initial_condition(num_psites)
    lower_bound = 1e-6
    for value in init_cond:
        assert value >= lower_bound

def verify_initial_condition_handles_zero_psites():
    num_psites = 0
    init_cond = initial_condition(num_psites)
    assert isinstance(init_cond, list)
    assert len(init_cond) == 2

def verify_initial_condition_raises_error_for_negative_psites():
    with pytest.raises(ValueError):
        initial_condition(-1)