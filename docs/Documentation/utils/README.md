# Utils

The `utils` module provides a set of helper scripts to streamline data processing, table generation, file organization, and report creation for the project. These utilities ensure consistent handling of outputs, formatting, and organization across the pipeline.

## Scripts Overview

### `display.py`
- **Purpose**: Handles data loading, output directory management, and report generation.
- **Key Features**:
  - Ensures output directories exist.
  - Loads data from Excel files.
  - Merges observed and estimated data for analysis.
  - Saves results to Excel with multiple sheets for parameters, errors, PCA, t-SNE, and sensitivity analysis.
  - Generates a global HTML report summarizing results with plots and tables.

### `tables.py`
- **Purpose**: Generates hierarchical tables for alpha and beta values and saves them in LaTeX and CSV formats.
- **Key Features**:
  - Processes alpha and beta values from Excel files.
  - Creates hierarchical tables with multi-index columns for easy comparison.
  - Saves tables as LaTeX and CSV files for further analysis.
  - Generates a master LaTeX file to include all individual tables.

### `latexit.py`
- **Purpose**: Converts Excel data and PNG plots into LaTeX tables and figures for documentation.
- **Key Features**:
  - Processes Excel sheets and generates LaTeX tables.
  - Converts PNG plots into LaTeX figure blocks.
  - Outputs a structured LaTeX file for integration into larger documents.

## Outputs
- **Excel Files**: Organized results with multiple sheets for parameters, errors, and analysis.
- **LaTeX Files**: Tables and figures for documentation.
- **CSV Files**: Processed data tables for further analysis.
- **HTML Reports**: Interactive summaries of results with plots and tables.