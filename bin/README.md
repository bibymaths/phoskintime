# ODE Estimation - Entry Point

This tool performs parallelized parameter estimation for phosphorylation dynamics models using time series data. It
reads input from an Excel sheet, processes each gene using a custom ODE-based fitting routine, and outputs results,
organized files, and an HTML report.

## Input Format

Excel file with sheet name `Estimated`, and columns:

- `Gene` (str)
- `Psite` (str)
- `x1` to `x14` (float): Time series data points

## Configuration

Configuration is passed via command-line arguments and processed using `config/config.py`. Key parameters include:

- `input_excel`: Path to Protein-Kinase data 
- `input_excel_rna`: Path to mRNA data 
- `bootstraps` : Number of bootstrap iterations
- `A-bound` : Bounds for mRNA production rate 
- `B-bound` : Bounds for mRNA degradation rate 
- `C-bound` : Bounds for protein production rate 
- `D-bound` : Bounds for protein degradation rate 
- `S-bound` : Bounds for phosphorylation rate 
- `D-bound` : Bounds for dephosphorylation rate

## Output

- Fitted results saved as Excel in `OUT_RESULTS_DIR`
- HTML report generated in `OUT_DIR/report.html`
- Logs and intermediate files organized in `OUT_DIR`

## Notes

- By default, only the **first gene** is processed (for testing).
- Make sure required columns exist in your input Excel sheet.