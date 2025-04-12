import pandas as pd
import numpy as np
import mygene, os, concurrent.futures
from tqdm import tqdm

# Directory where the raw data files are located
base_dir = "raw"

###############################
# 1. Process CollectTRI File  #
###############################
def process_collecttri():
    # Load CollecTRI.csv and keep only the source and target columns
    df = pd.read_csv(os.path.join(base_dir, "CollecTRI.csv"))
    df_readable = df[['source_genesymbol', 'target_genesymbol']].rename(
        columns={'source_genesymbol': 'Source', 'target_genesymbol': 'Target'}
    )
    # Remove rows with NaN, empty strings or whitespace, and drop duplicates
    df_readable = df_readable.dropna()
    df_readable = df_readable[df_readable['Source'].str.strip() != '']
    df_readable = df_readable[df_readable['Target'].str.strip() != '']
    df_readable = df_readable.drop_duplicates()

    # Load HitGenes.xlsx (first sheet) and filter by the 'gene' column
    df_genes = pd.read_excel(os.path.join(base_dir, "HitGenes.xlsx"), sheet_name=0)
    df_genes = df_genes[['gene']].rename(columns={'gene': 'Source'})
    df_genes = df_genes.dropna()
    df_genes = df_genes[df_genes['Source'].str.strip() != '']
    df_genes = df_genes.drop_duplicates()

    # Keep only interactions where Source is present in HitGenes.xlsx
    df_readable = df_readable[df_readable['Source'].isin(df_genes['Source'])]

    # Save the cleaned interactions to input4.csv
    df_readable.to_csv("input4.csv", index=False)
    print("Saved TF-mRNA interactions to input4.csv")

####################################
# Utility: Format Site Information #
####################################
def format_site(site):
    if pd.isna(site) or site == '':
        return ''
    if '_' in site:
        before, after = site.split('_', 1)
        return before.upper() + '_' + after
    else:
        return site.upper()

##############################################
# 2. Process MS Gaussian (predict_mean only) #
##############################################
def process_msgauss():
    # Load the MS_Gaussian_updated_09032023.csv file
    df = pd.read_csv(os.path.join(base_dir,"MS_Gaussian_updated_09032023.csv"))
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
    # Filter the empty Psite values and keep only ones which start with Y_, S_, and T_
    kinopt_df = pivot_df[pivot_df['Psite'].str.startswith(('Y_', 'S_', 'T_'))]
    # Save the filtered time series to input1.csv
    kinopt_df.to_csv("input1.csv", index=False)
    print("Saved MS Gaussian (predict_mean) time series to input1.csv")

######################################################
# 3. Process MS Gaussian with Standard Deviation Data #
######################################################
def process_msgauss_std():
    df = pd.read_csv(os.path.join(base_dir,"MS_Gaussian_updated_09032023.csv"))
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
    # Filter the empty Psite values and keep only ones which start with Y_, S_, and T_
    result = result[result['Psite'].str.startswith(('Y_', 'S_', 'T_'))]
    # Save to input1_wstd.csv
    result.to_csv("input1_wstd.csv", index=False)
    print("Saved MS Gaussian time-series with standard deviations to input1_wstd.csv")

#####################################
# 4. Process Rout Limma Data File   #
#####################################
def process_routlimma():
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

#####################################################
# 5. Update Gene Symbols using mygene and TQDM     #
#####################################################
def update_gene_symbols(filename):
    """
    Reads a CSV file, updates the GeneID column by mapping to gene/protein symbols using mygene,
    and writes the updated DataFrame back to the same file.
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

    # print("Query results preview:")
    # print(query_results.head())

    # Filter out not found results if available.
    if 'notfound' in query_results.columns:
        query_results = query_results[query_results['notfound'] != True]

    # Build the mapping dictionary from queried GeneIDs to gene symbols.
    mapping = query_results['symbol'].to_dict()

    def map_geneid(geneid):
        return mapping.get(str(geneid), geneid)

    # Use ThreadPoolExecutor with tqdm to parallelize mapping.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(map_geneid, df["GeneID"]), total=len(df["GeneID"]),
                            desc=f"Processing {filename}"))

    df["GeneID"] = results
    df.to_csv(filename, index=False)
    print(f"Updated gene symbols in {filename}") 
       
#####################################################
# 6. Reduces the processed CollecTRI file to TFs    #
# present in phosphorylation and kinase interaction #
# data.                                             #
#####################################################
def filter_tf_by_phospho(input4_path="input4.csv",
                         input2_path="input2.csv"):
    # Load input4.csv (which contains the Source and Target columns)
    df_input4 = pd.read_csv(input4_path)

    # Load input2.csv (which should contain a column named 'GeneID')
    df_input2 = pd.read_csv(input2_path)

    # Create a set of GeneID values from input2 for fast lookup
    gene_ids = set(df_input2["GeneID"].dropna().astype(str))

    # Filter rows in input4:
    # Keep rows if either the Source or Target is found in the gene_ids set.
    filtered_df = df_input4[
        df_input4["Source"].astype(str).isin(gene_ids) |
        df_input4["Target"].astype(str).isin(gene_ids)
    ]

    # Save the filtered DataFrame to a new CSV file (or overwrite input4.csv if preferred)
    filtered_df.to_csv(input4_path, index=False)

#####################################################
# 7. Move processed files to respective directories #
#####################################################
def move_processed_files():
    # Create a new directory if it doesn't exist
    tf_data_dir = "../tfopt/data"
    kin_data_dir = "../kinopt/data"

    if not os.path.isdir(tf_data_dir) and os.path.exists(tf_data_dir):
        os.makedirs(tf_data_dir)
        os.makedirs(kin_data_dir)

    # List of files to move
    kinopt_files = [
        "input1.csv",
    ]
    tfopt_files = [
        "input1.csv",
        # "input1_wstd.csv",
        "input3.csv",
        "input4.csv"
    ]

    # Move each file for kinopt and tfopt to theire respective directories
    for f in kinopt_files:
        if os.path.exists(f):
            os.rename(f, os.path.join(kin_data_dir, f))
            print(f"Moved {f} to {kin_data_dir}")
        else:
            print(f"{f} does not exist in the current directory.")

    for f in tfopt_files:
        if os.path.exists(f):
            os.rename(f, os.path.join(tf_data_dir, f))
            print(f"Moved {f} to {tf_data_dir}")
        else:
            print(f"{f} does not exist in the current directory.")


#########################
# Run All Processing    #
#########################
if __name__ == "__main__":
    # Process files
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

    # 6. Filter TFs from input4.csv based on phosphorylation and kinase interaction data
    filter_tf_by_phospho()

    # 7. Move processed files to the appropriate directories
    # You need to manually put input2.csv in ./kinopt/data/ (if not already there)
    move_processed_files()