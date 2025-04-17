
import pytest
from steady.initdist import initial_condition as initial_condition_dist
from steady.initsucc import initial_condition as initial_condition_succ
from steady.initrand import initial_condition as initial_condition_rand


def run_common_tests(initial_condition_func, model_name, num_psites):
    # Get the initial condition and test its return type and length.
    init_cond = initial_condition_func(num_psites)
    assert isinstance(init_cond, list), f"{model_name} did not return a list"
    expected_length = num_psites + 2 if num_psites >= 0 else None
    assert len(init_cond) == expected_length, f"{model_name} expected list length {expected_length}"

    # Check that all values meet the lower bound.
    lower_bound = 1e-6
    for value in init_cond:
        assert value >= lower_bound, f"{model_name} returned a value below {lower_bound}"


def test_initdist_valid_conditions():
    num_psites = 3
    run_common_tests(initial_condition_dist, "initdist", num_psites)


def test_initsucc_valid_conditions():
    num_psites = 3
    run_common_tests(initial_condition_succ, "initsucc", num_psites)


def test_initrand_valid_conditions():
    num_psites = 3
    run_common_tests(initial_condition_rand, "initrand", num_psites)


def test_initdist_zero_psites():
    init_cond = initial_condition_dist(0)
    assert isinstance(init_cond, list)
    assert len(init_cond) == 2


def test_initsucc_zero_psites():
    init_cond = initial_condition_succ(0)
    assert isinstance(init_cond, list)
    assert len(init_cond) == 2


def test_initrand_zero_psites():
    init_cond = initial_condition_rand(0)
    assert isinstance(init_cond, list)
    assert len(init_cond) == 2


def test_initdist_negative_psites():
    with pytest.raises(ValueError):
        initial_condition_dist(-1)


def test_initsucc_negative_psites():
    with pytest.raises(ValueError):
        initial_condition_succ(-1)


def test_initrand_negative_psites():
    with pytest.raises(ValueError):
        initial_condition_rand(-1)