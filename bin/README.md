# ODE Estimation - Entry Point

This tool performs parallelized parameter estimation for phosphorylation dynamics models using time series data. It reads input from an Excel sheet, processes each gene using a custom ODE-based fitting routine, and outputs results, organized files, and an HTML report.

## Features

- Parallel processing of multiple genes using `ProcessPoolExecutor`
- Configurable parameter bounds, fixed values, and bootstrap settings
- Logging and error handling
- Report and result generation

## Input Format

Excel file with sheet name `Estimated`, and columns:

- `Gene` (str)
- `Psite` (str)
- `x1` to `x14` (float): Time series data points

## Configuration

Configuration is passed via command-line arguments and processed using `config/config.py`. Key parameters include:

- `input_excel`: Path to Excel file
- `bounds`, `fixed_params`: Estimation constraints
- `bootstraps`, `time_fixed`: Optional features
- `profile_start`, `profile_end`, `profile_step`: Optional profiling range

## Output

- Fitted results saved as Excel in `OUT_RESULTS_DIR`
- HTML report generated in `OUT_DIR/report.html`
- Logs and intermediate files organized in `OUT_DIR`

## Notes

- By default, only the **first gene** is processed (for testing).
- Make sure required columns exist in your input Excel sheet.