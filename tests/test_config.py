from config.config import extract_config


def test_extract_config_defaults():
    class Args:
        A_bound = (0, 20)
        B_bound = (0, 20)
        C_bound = (0, 20)
        D_bound = (0, 20)
        Ssite_bound = (0, 20)
        Dsite_bound = (0, 20)

        fix_A = None
        fix_B = 1
        fix_C = 1
        fix_D = 1
        fix_Ssite = None
        fix_Dsite = 1

        fix_t = ''
        bootstraps = 0
        profile_start = 0
        profile_end = 100
        profile_step = 10

        input_excel = "tests/mock.xlsx"

    config = extract_config(Args())

    bounds = config['bounds']
    fixed = config['fixed_params']
    tfixed = config['time_fixed']

    assert bounds["A"] == (0, 20)
    assert fixed["B"] == 1
    assert tfixed == {}
