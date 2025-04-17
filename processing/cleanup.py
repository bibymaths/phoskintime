import shutil

import pandas as pd
import numpy as np
import mygene, os, concurrent.futures
from tqdm import tqdm

# Directory where the raw data files should be located
base_dir = "raw"

def process_collecttri():
    """
    Processes the CollecTRI file to clean and filter mRNA-TF interactions.
    Removes complex interactions, filters by target genes, and saves the result.
    """

    # Load CollecTRI.csv and keep only the source and target columns
    df = pd.read_csv(os.path.join(base_dir, "CollecTRI.csv"))

    # Remove thr rows that starts with 'COMPLEX' in the source column
    # Model doesn't support complex interactions, ony single mRNA
    df = df[~df['source'].str.startswith('COMPLEX')]

    df_readable = df[['source_genesymbol', 'target_genesymbol']].rename(
        columns={'source_genesymbol': 'Source', 'target_genesymbol': 'Target'}
    )
    # Remove rows with NaN, empty strings or whitespace, and drop duplicates
    df_readable = df_readable.dropna()
    df_readable = df_readable[df_readable['Source'].str.strip() != '']
    df_readable = df_readable[df_readable['Target'].str.strip() != '']
    df_readable = df_readable.drop_duplicates()

    # Load phospho-kinase interaction data and search CollectTRI for same TFs
    df_genes = pd.read_csv(os.path.join(base_dir, "input2.csv"))
    df_genes = df_genes[['GeneID']].rename(columns={'GeneID': 'Target'})
    df_genes = df_genes.dropna()
    df_genes = df_genes[df_genes['Target'].str.strip() != '']
    df_genes = df_genes.drop_duplicates()

    # Keep only interactions where Target (TFs) is present in input2.csv
    df_readable = df_readable[df_readable['Target'].isin(df_genes['Target'])]

    # Save the cleaned mRNA - TFs interactions to input4.csv
    df_readable.to_csv("input4.csv", index=False)

    # Copy the input2.csv file to the current directory using shutil
    shutil.copy(os.path.join(base_dir, "input2.csv"), "input2.csv")

    print("Saved TF-mRNA interactions to input4.csv")

def format_site(site):
    """
    Formats a phosphorylation site string.

    If the input is NaN or an empty string, returns an empty string.
    If the input contains an underscore ('_'), splits the string into two parts,
    converts the first part to uppercase, and appends the second part unchanged.
    Otherwise, converts the entire string to uppercase.

    Args:
        site (str): The phosphorylation site string to format.

    Returns:
        str: The formatted phosphorylation site string.
    """
    if pd.isna(site) or site == '':
        return ''
    if '_' in site:
        before, after = site.split('_', 1)
        return before.upper() + '_' + after
    else:
        return site.upper()

def process_msgauss():
    """
    Processes the MS Gaussian data file to generate time series data.

    This function performs the following steps:
    1. Loads the `MS_Gaussian_updated_09032023.csv` file.
    2. Computes the transformed values as 2^(predict_mean).
    3. Pivots the data to create a time series for each (GeneID, Psite) pair.
    4. Renames the time point columns to `x1` to `x14` and formats the `Psite` column.
    5. Saves the cleaned time series to `input1.csv`.
    6. Filters the data to keep only rows where `Psite` starts with `Y_`, `S_`, `T_`, or is empty.
    7. Saves the filtered time series to `input1.csv`.

    Outputs:
        - `input1.csv`: Cleaned and filtered time series data.

    Prints:
        - A message indicating that the time series data has been saved.

    Raises:
        FileNotFoundError: If the input file is not found in the specified directory.
    """
    # Load the MS_Gaussian_updated_09032023.csv file
    df = pd.read_csv(os.path.join(base_dir, "MS_Gaussian_updated_09032023.csv"))

    df['Psite'] = df['site'].fillna('').astype(str)

    # Compute 2^(predict_mean)
    df['predict_trans'] = 2 ** df['predict_mean']

    # Pivot so that each (GeneID, Psite) pair has time series values per unit_time (0-13)
    pivot_df = df.pivot_table(
        index=['GeneID', 'Psite'],
        columns='unit_time',
        values='predict_trans',
        aggfunc='first'
    ).reset_index()

    # Rename time point columns to x1 to x14 and format the Psite column
    new_names = {i: f'x{i + 1}' for i in range(14)}
    pivot_df.rename(columns=new_names, inplace=True)
    pivot_df['Psite'] = pivot_df['Psite'].apply(format_site)

    # Save the cleaned time series to input1.csv
    pivot_df.to_csv("input1.csv", index=False)

    # Filter to keep only rows where 'Psite' starts with Y_, S_, T_, or is empty
    kinopt_df = pivot_df[
        pivot_df['Psite'].str.startswith(('Y_', 'S_', 'T_')) | (pivot_df['Psite'] == '')]

    # Save the filtered time series to input1.csv
    kinopt_df.to_csv("input1.csv", index=False)

    print("Saved MS Gaussian (predict_mean) time series to input1.csv")


def process_msgauss_std():
    """
    Processes the MS Gaussian data file to compute transformed means and standard deviations.

    This function performs the following steps:
    1. Loads the `MS_Gaussian_updated_09032023.csv` file.
    2. Computes transformed values as 2^(predict_mean) and propagates errors using the formula:
       sigma_y = 2^(x) * ln(2) * sigma_x.
    3. Pivots the data to create time series for each (GeneID, Psite) pair for both means and standard deviations.
    4. Merges the pivoted data for means and standard deviations.
    5. Filters the data to keep only rows where `Psite` starts with `Y_`, `S_`, `T_`, or is empty.
    6. Saves the resulting data to `input1_wstd.csv`.

    Outputs:
        - `input1_wstd.csv`: Cleaned time series data with transformed means and standard deviations.

    Prints:
        - A message indicating that the time series data with standard deviations has been saved.

    Raises:
        FileNotFoundError: If the input file is not found in the specified directory.
    """
    df = pd.read_csv(os.path.join(base_dir, "MS_Gaussian_updated_09032023.csv"))
    df['Psite'] = df['site'].fillna('').astype(str)
    df['predict_trans'] = 2 ** df['predict_mean']

    # Error propagation: sigma_y = 2^(x) * ln(2) * sigma_x
    df['predict_trans_std'] = df['predict_trans'] * np.log(2) * df['predict_std']

    # Pivot for the transformed means
    pivot_trans = df.pivot_table(
        index=['GeneID', 'Psite'],
        columns='unit_time',
        values='predict_trans',
        aggfunc='first'
    ).reset_index()

    pivot_trans = pivot_trans.rename(columns={i: f'x{i + 1}' for i in range(14)})

    # Pivot for the transformed standard deviations
    pivot_std = df.pivot_table(
        index=['GeneID', 'Psite'],
        columns='unit_time',
        values='predict_trans_std',
        aggfunc='first'
    ).reset_index()
    pivot_std = pivot_std.rename(columns={i: f'x{i + 1}_std' for i in range(14)})

    # Merge the two pivot tables and format Psite
    result = pd.merge(pivot_trans, pivot_std, on=['GeneID', 'Psite'])
    result['Psite'] = result['Psite'].apply(format_site)

    # Filter to keep only rows where 'Psite' starts with Y_, S_, T_, or is empty
    result = result[result['Psite'].str.startswith(('Y_', 'S_', 'T_')) | (result['Psite'] == '')]

    # Save to input1_wstd.csv
    result.to_csv("input1_wstd.csv", index=False)

    print("Saved MS Gaussian time-series with standard deviations to input1_wstd.csv")

def process_routlimma():
    """
    Processes the Rout Limma table to generate time series data for mRNA.

    This function performs the following steps:
    1. Loads the `Rout_LimmaTable.csv` file.
    2. Selects specific columns related to time points and conditions.
    3. Renames the selected columns to a standardized format (`x1` to `x9`).
    4. Converts the values in the renamed columns using the formula `2^(value)`.
    5. Saves the resulting time series data to `input3.csv`.

    Outputs:
        - `input3.csv`: Cleaned and transformed time series data for mRNA.

    Prints:
        - A message indicating that the time series data has been saved.

    Raises:
        FileNotFoundError: If the input file `Rout_LimmaTable.csv` is not found in the specified directory.
    """
    # Load Rout_LimmaTable.csv and select desired columns
    df = pd.read_csv(os.path.join(base_dir, "Rout_LimmaTable.csv"))
    selected_cols = [
        'GeneID',
        'Min4vsCtrl', 'Min8vsCtrl', 'Min15vsCtrl', 'Min30vsCtrl',
        'Hr1vsCtrl', 'Hr2vsCtrl', 'Hr4vsCtrl', 'Hr8vsCtrl', 'Hr16vsCtrl'
    ]
    df_new = df[selected_cols]

    # Rename the condition columns to x1 to x9
    rename_mapping = {
        'Min4vsCtrl': 'x1',
        'Min8vsCtrl': 'x2',
        'Min15vsCtrl': 'x3',
        'Min30vsCtrl': 'x4',
        'Hr1vsCtrl': 'x5',
        'Hr2vsCtrl': 'x6',
        'Hr4vsCtrl': 'x7',
        'Hr8vsCtrl': 'x8',
        'Hr16vsCtrl': 'x9'
    }
    df_new = df_new.rename(columns=rename_mapping)

    # Convert each value in x1 to x9 as 2^(value)
    for col in rename_mapping.values():
        df_new[col] = 2 ** df_new[col]

    # Save the resulting time series to input3.csv
    df_new.to_csv("input3.csv", index=False)

    print("Saved Rout Limma time series - mRNA to input3.csv")

def update_gene_symbols(filename):
    """
    Updates the GeneID column in a CSV file by mapping GeneIDs to gene/protein symbols.

    This function reads a CSV file, queries the MyGeneInfo database to map GeneIDs to their
    corresponding gene/protein symbols, and writes the updated DataFrame back to the same file.

    Args:
        filename (str): The path to the CSV file to be updated. The file must contain a 'GeneID' column.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the 'GeneID' column is missing in the file.

    Outputs:
        - The input file is updated in place with the 'GeneID' column replaced by gene/protein symbols.

    Prints:
        - A progress bar indicating the mapping process.
        - A message confirming the update of gene symbols in the file.
    """
    df = pd.read_csv(filename)
    df['GeneID'] = df['GeneID'].astype(str)
    unique_gene_ids = list(df["GeneID"].unique())

    # Initialize MyGeneInfo client and query in bulk.
    mg = mygene.MyGeneInfo()
    query_results = mg.querymany(unique_gene_ids,
                                 scopes='ensembl.gene,entrezgene,symbol',
                                 species='human',
                                 as_dataframe=True)

    # Filter out not found results if available.
    if 'notfound' in query_results.columns:
        query_results = query_results[query_results['notfound'] != True]

    # Build the mapping dictionary from queried GeneIDs to gene symbols.
    mapping = query_results['symbol'].to_dict()

    def map_geneid(geneid):
        """
        Maps a single GeneID to its corresponding gene/protein symbol.

        Args:
            geneid (str): The GeneID to be mapped.

        Returns:
            str: The corresponding gene/protein symbol if found, otherwise the original GeneID.
        """
        return mapping.get(str(geneid), geneid)

    # Use ThreadPoolExecutor with tqdm to parallelize mapping.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(map_geneid, df["GeneID"]), total=len(df["GeneID"]),
                            desc=f"Processing {filename}"))

    df["GeneID"] = results
    df.to_csv(filename, index=False)
    print(f"Updated gene symbols in {filename}")

def move_processed_files():
    """
    Moves or copies processed files to their respective directories.

    This function organizes processed files into specific directories for
    transcription factor optimization (TFOpt) and kinase optimization (KinOpt).
    It creates the target directories if they do not exist and moves or copies
    the files based on whether they have already been moved.

    Directories:
        - `../tfopt/data`: Target directory for TFOpt files.
        - `../kinopt/data`: Target directory for KinOpt files.

    Files:
        - KinOpt files: ["input1.csv", "input2.csv"]
        - TFOpt files: ["input1.csv", "input3.csv", "input4.csv"]

    Behavior:
        - If a file has already been moved, it is copied to the target directory.
        - If a file has not been moved, it is moved to the target directory.
        - If a file does not exist, a message is printed.

    Prints:
        - Messages indicating whether files were moved, copied, or not found.
    """

    # Create a new directory if it doesn't exist
    tf_data_dir = "../tfopt/data"
    kin_data_dir = "../kinopt/data"

    os.makedirs(tf_data_dir, exist_ok=True)
    os.makedirs(kin_data_dir, exist_ok=True)

    # List of files to move
    kinopt_files = [
        "input1.csv",
        "input2.csv"
    ]
    tfopt_files = [
        "input1.csv",
        # "input1_wstd.csv",
        "input3.csv",
        "input4.csv"
    ]

    # Track files already moved so we copy them next time
    moved_files = set()

    # Move or copy files to their respective directories
    for file_list, target_dir in [(kinopt_files, kin_data_dir), (tfopt_files, tf_data_dir)]:
        for f in file_list:
            if os.path.exists(f):
                target_path = os.path.join(target_dir, f)
                if f in moved_files:
                    shutil.copy(target_path, os.path.join(target_dir, f))
                    print(f"Copied {f} to {target_dir}")
                else:
                    shutil.move(f, target_path)
                    moved_files.add(f)
                    print(f"Moved {f} to {target_dir}")
            else:
                print(f"{f} does not exist in the current directory or has already been moved.")


if __name__ == "__main__":

    # 1. Process CollectTRI - TF interactions with mRNA
    process_collecttri()

    # 2. Process MS Gaussian - Time series data with proteins and its phosphorylation sites
    process_msgauss()

    # 3. Process MS Gaussian to keep standard deviation data for each time point
    # Possible Usage: Plotting error bars, error propagation, etc.
    process_msgauss_std()

    # 4. Process Rout Limma - Time series data mRNA
    process_routlimma()

    # List of files to update
    # (all except input2.csv and input4.csv) - has symbols already
    replace_ID_to_Symbol = [
        "input1.csv",
        "input1_wstd.csv",
        "input3.csv"
    ]

    # 5. Update gene symbols in the specified files
    for file in replace_ID_to_Symbol:
        update_gene_symbols(file)

    # 6. Move processed files to the appropriate directories
    # You need to manually put input2.csv in ./kinopt/data/ (if not already there)
    move_processed_files()

""" 
These IDs, cannot be converted to Symbols in MS Gaussian:

4 input query terms found no hit:	['55747', '283331', '729269', '100133171'] 
"""