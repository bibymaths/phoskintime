import numpy as np
import itertools

def apply_knockout(base_params: np.ndarray,
                   knockout_targets: dict,
                   num_psites: int) -> np.ndarray:
    """
    Return a modified parameter vector simulating in silico knockouts.

    knockout_targets can include keys:
      - 'transcription': bool (zero A)
      - 'translation': bool (zero C)
      - 'phosphorylation': bool or list of site indices (zero S_rates)

    :param base_params: original parameters (A,B,C,D, S_rates..., D_rates...)
    :param knockout_targets: dict specifying which processes to knock out
    :param num_psites: number of phosphorylation sites
    :return: new params array with specified knockouts
    """
    params = base_params.copy()
    # Transcription knockout
    if knockout_targets.get('transcription', False):
        params[0] = 0.0
    # Translation knockout
    if knockout_targets.get('translation', False):
        params[2] = 0.0
    # Phosphorylation knockout
    if 'phosphorylation' in knockout_targets:
        k = knockout_targets['phosphorylation']
        start = 4
        end = start + num_psites
        if isinstance(k, bool) and k:
            params[start:end] = 0.0
        elif isinstance(k, (list, tuple)):
            for idx in k:
                if 0 <= idx < num_psites:
                    params[start + idx] = 0.0
    return params

def generate_knockout_combinations(num_psites: int):
    """
    Generate all possible knockout combinations.
    """
    combinations = []
    transcription_options = [False, True]
    translation_options = [False, True]
    # Phosphorylation options: False (none), True (all), individual sites
    phosphorylation_options = [False, True] + [[i] for i in range(num_psites)]

    for trans, transl, phospho in itertools.product(transcription_options, translation_options, phosphorylation_options):
        knockout = {
            'transcription': trans,
            'translation': transl,
            'phosphorylation': phospho,
        }
        combinations.append(knockout)
    return combinations

