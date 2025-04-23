# Utils

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