# Knockout

The `knockout` module provides utilities for simulating in silico knockouts of biological processes and generating all possible knockout combinations. It is designed to work with parameter vectors representing biological systems.

## Features

- **Apply Knockouts:**  
  Modify parameter vectors to simulate knockouts for transcription, translation, or phosphorylation processes.

- **Generate Knockout Combinations:**  
  Create all possible combinations of knockouts, including individual phosphorylation site knockouts.

## Functions

### `apply_knockout`
Simulates knockouts by modifying a given parameter vector.

- **Parameters:**
  - `base_params` (`np.ndarray`): Original parameter vector.
  - `knockout_targets` (`dict`): Specifies processes to knock out (`transcription`, `translation`, `phosphorylation`).
  - `num_psites` (`int`): Number of phosphorylation sites.

- **Returns:**  
  A modified parameter vector with the specified knockouts applied.

### `generate_knockout_combinations`
Generates all possible combinations of knockouts for transcription, translation, and phosphorylation.

- **Parameters:**
  - `num_psites` (`int`): Number of phosphorylation sites.

- **Returns:**  
  A list of dictionaries, each representing a unique knockout combination.