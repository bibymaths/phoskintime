

# Utils Module 

The **utils** module provides a collection of helper functions that streamline data handling, output formatting, file management, and report/table generation throughout the PhosKinTime package. These utilities ensure that results from parameter estimation, sensitivity analysis, and other computations are organized, saved, and displayed consistently.

## Module Structure

The **utils** module is organized into the following components:

### 1. Display Utilities (`display.py`)

This submodule includes functions for:

- **Directory Management:**  
  - `ensure_output_directory(directory)`: Creates the specified directory if it does not exist.

- **Data Loading:**  
  - `load_data(excel_file, sheet="Estimated Values")`: Loads and returns data from an Excel file.

- **Formatting:**  
  - `format_duration(seconds)`: Converts a duration in seconds to a human-readable format (seconds, minutes, or hours).

- **Result Saving:**  
  - `save_result(results, excel_filename)`: Saves a list of result dictionaries to an Excel file with separate sheets for each gene’s parameters, profiles, and error summaries.

- **Report Generation:**  
  - `create_report(results_dir, output_file="report.html")`: Generates a global HTML report by aggregating plots and data tables from gene-specific result folders.
  
- **File Organization:**  
  - `organize_output_files(*directories)`: Organizes output files by moving gene-specific files into subfolders and grouping remaining files into a "General" folder.

### 2. Table Utilities (`tables.py`)

This submodule provides functions for generating and saving data tables:

- **Table Generation:**  
  - `generate_tables(xlsx_file_path)`: Loads alpha and beta values from an Excel file, pivots the data, and creates hierarchical tables combining both sets of values.

- **Table Saving:**  
  - `save_tables(tables, output_dir)`: Saves each generated hierarchical table as both a LaTeX file and a CSV file, using a naming convention based on protein and phosphorylation site.

- **Master Table Creation:**  
  - `save_master_table(folder="latex", output_file="latex/all_tables.tex")`: Generates a master LaTeX file that includes all the individual table files from a specified folder.

## Usage Example

Below is a sample code snippet that demonstrates how to use the utilities:

```python
from utils.display import ensure_output_directory, load_data, format_duration, save_result, create_report, organize_output_files
from utils.tables import generate_tables, save_tables, save_master_table

# Create the output directory if it doesn't exist
output_dir = "./results"
ensure_output_directory(output_dir)

# Load experimental data from an Excel file
data = load_data("data/optimization_results.xlsx")

# Format a duration (e.g., 125 seconds becomes "2.08 min")
print(format_duration(125))

# Save a list of result dictionaries to an Excel file
results = [
    {"gene": "GeneX", "param_df": ...},  # Replace "..." with your DataFrame
    {"gene": "GeneY", "param_df": ...}
]
save_result(results, "results/combined_results.xlsx")

# Generate a global HTML report from the results directory
create_report(output_dir, "report.html")

# Organize output files into gene-specific folders
organize_output_files(output_dir)

# Generate and save hierarchical tables from an Excel file
tables = generate_tables("data/optimization_results.xlsx")
save_tables(tables, output_dir)
save_master_table("latex", "latex/all_tables.tex")
```

## Conclusion

The **utils** module is a critical component of the PhosKinTime package. It provides robust functions for ensuring directories exist, loading and formatting data, saving results, generating comprehensive HTML reports, and producing publication-ready tables. These utilities facilitate a seamless workflow from data analysis to the presentation of results, making the package easier to use and customize.

