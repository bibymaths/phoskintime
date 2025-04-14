# Data Cleanup and Processing Workflow

This project standardizes raw data files for downstream analysis in kinase and TF-mRNA optimization pipelines. The processing script (`cleanup.py`) performs data cleanup, transformation, and then moves the processed files into designated directories.

## Directory Structure

The raw files should be stored under the `raw/` directory with the following structure:

```
raw
├── CollecTRI.csv
├── HitGenes.xlsx
├── input2.csv
├── MS_Gaussian_updated_09032023.csv
└── Rout_LimmaTable.csv
```

*Note:* The file `HitGenes.xlsx` is currently not processed by the script as it irrelevant in the pipeline.  
It is just there as a placeholder if one wants to use velocity data in some scenario.   


## Processing Steps

The `cleanup.py` script performs the following:

1. **Processing CollecTRI.csv**
   - Reads and filters out rows where the `source` column starts with "COMPLEX".
   - Keeps only the `source_genesymbol` and `target_genesymbol` columns (renamed to "Source" and "Target").
   - Removes rows with missing or blank entries and duplicates.
   - Retains interactions only if the target gene appears in `input2.csv`.
   - Saves the cleaned TF-mRNA interactions as `input4.csv` and copies `input2.csv` for further use.

2. **Processing MS_Gaussian_updated_09032023.csv**
   - Transforms the `predict_mean` column by computing `2^(predict_mean)`.
   - Pivots the data so each `(GeneID, Psite)` pair has a time series (columns `x1` to `x14`).
   - Formats the phosphorylation site (Psite) values and filters to keep only rows starting with `Y_`, `S_`, or `T_`.
   - Saves the resulting time series data as `input1.csv`.

3. **Processing Standard Deviation Data for MS Gaussian**
   - Computes the standard deviation transformation using the error propagation formula as we are comverting logFC to FC in `predict_mean` column so the `predict_std` column needs to be transformed as well:
     
     ```
     sigma_trans = 2^(predict_mean) * ln(2) * predict_std
     ```
     
   - Pivots and merges the mean and standard deviation data.
   - Filters and saves the processed data as `input1_wstd.csv`.

4. **Processing Rout_LimmaTable.csv**
   - Extracts specific condition columns and renames them to `x1` to `x9`.
   - Converts the values using the transformation `2^(value)`.
   - Saves the time series data as `input3.csv`.

5. **Updating Gene Symbols**
   - Uses the `mygene` package to update the gene IDs to standard gene symbols across key processed files (`input1.csv`, `input1_wstd.csv`, and `input3.csv`).

6. **Moving Processed Files**
   - **Kinase-Phosphrylation Optimization (kinopt):**
     - Moves `input1.csv` and `input2.csv` to `../kinopt/data`.
   - **TF-mRNA Optimization (tfopt):**
     - Moves `input1.csv`, `input3.csv`, and `input4.csv` to `../tfopt/data`.

*Note:* The file `input1.csv` is used by both pipelines. Addtionally, the following IDs, cannot be converted to Symbols in MS Gaussian:
 
```
4 input query terms found no hit:	['55747', '283331', '729269', '100133171'] 
``` 

## Setup and Running the Pipeline

1. **Directory Setup:**
   - Create a directory named `raw` in your project root.
   - Place the raw files (`CollecTRI.csv`, `HitGenes.xlsx`, `input2.csv`, `MS_Gaussian_updated_09032023.csv`, `Rout_LimmaTable.csv`) into the `raw` folder.

2. **Install Dependencies:**
   - Ensure the project is set up in a Python environment.

3. **Execution:**
   - Run the cleanup script from your project root:
     ```bash
     python cleanup.py
     ```