import numpy as np
import itertools


def _apply_knockout(base_params: np.ndarray,
                    knockout_targets: dict,
                    num_psites: int) -> np.ndarray:
    """
    Return a modified parameter vector simulating in silico knockouts.

    Args:
        base_params (np.ndarray): Original parameter vector.
        knockout_targets (dict): Dictionary with knockout targets.
        num_psites (int): Number of phosphorylation sites.
    Returns:
        np.ndarray: Modified parameter vector with knockouts applied.
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


def _generate_knockout_combinations(num_psites: int):
    """
    Generate all possible knockout combinations.

    Args:
        num_psites (int): Number of phosphorylation sites.
    Returns:
        list: List of dictionaries representing knockout combinations.
    """
    combinations = []
    transcription_options = [False, True]
    translation_options = [False, True]
    # Phosphorylation options: False (none), True (all), individual sites
    phosphorylation_options = [False, True] + [[i] for i in range(num_psites)]

    for trans, transl, phospho in itertools.product(transcription_options, translation_options,
                                                    phosphorylation_options):
        knockout = {
            'transcription': trans,
            'translation': transl,
            'phosphorylation': phospho,
        }
        combinations.append(knockout)
    return combinations
