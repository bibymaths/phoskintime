from phoskintime.config.config import extract_config

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
    bounds, fixed, tfixed = extract_config(Args())
    assert bounds["A"] == (0, 20)
    assert fixed["B"] == 1
    assert tfixed == {}
