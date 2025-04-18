import os
import pandas as pd

def map_optimization_results(file_path):
    """
    Reads the TF-mRNA optimization results from an Excel file and maps mRNA to each TF.

    This function processes the 'Alpha Values' sheet of the provided Excel file to extract
    non-zero optimization results, groups mRNA by TF, and merges the results with additional
    data from a CSV file containing TF, Psite, and Kinase information. The final DataFrame
    is cleaned and formatted for further analysis.

    Args:
        file_path (str): The path to the Excel file containing TF-mRNA optimization results.

    Returns:
        pd.DataFrame: A DataFrame containing the mapped TF, mRNA, Psite, and Kinase information.
    """
    # Read the Excel file
    tfopt_file = pd.ExcelFile(file_path)

    # Read the 'Alpha Values' sheet
    df = pd.read_excel(tfopt_file, sheet_name='Alpha Values')

    # Filter the DataFrame for non-zero values
    non_zero_df = df[df['Value'] != 0]

    # Filter for positive values
    # non_zero_df = df[df['Value'] > 0]

    # Take all values without filtering
    # non_zero_df = df

    # Filter very small values out
    # non_zero_df = non_zero_df[non_zero_df['Value'] > 0.0001]

    # Extract the mRNA for each TF where Values was not zero
    result = non_zero_df[['TF', 'mRNA']]

    # Display value along with TF and mRNA
    # result['Value'] = str(non_zero_df['Value'])

    result = result.groupby('TF').agg(lambda x: ', '.join(x)).reset_index()

    # List number of mRNA for each TF
    # result['mRNA_Count'] = result['mRNA'].apply(lambda x: len(x.split(',')))

    # Read another csv file which has TF aka GeneID and Psites and Kinases to merge with the result
    kinopt_file = pd.read_csv('raw/input2.csv')
    df2 = kinopt_file.rename(columns={'GeneID': 'TF'})
    merged_df = pd.merge(result, df2, on='TF', how='left')

    # Remove {} from the Kinases column with format {kinase1, kinase2, kinase3}
    merged_df['Kinase'] = merged_df['Kinase'].astype(str)
    merged_df['Kinase'] = merged_df['Kinase'].str.replace(r'{', '', regex=True)
    merged_df['Kinase'] = merged_df['Kinase'].str.replace(r'}', '', regex=True)

    # Remove any extra spaces
    merged_df['Kinase'] = merged_df['Kinase'].str.replace(r'\s+', '', regex=True)

    # Delete the rows where Psite doesn't match Psite in input2
    merged_df = merged_df[merged_df['Psite'].isin(df2['Psite'])]

    # Empty cells in mRNA column if they are repeateed in the next row
    merged_df['mRNA'] = merged_df['mRNA'].where(merged_df['mRNA'] != merged_df['mRNA'].shift(), '')

    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.reset_index(drop=True)

    return merged_df

if __name__ == "__main__":
    file_path = '../data/tfopt_results.xlsx'  # Replace with your file path
    # Call the function to map optimization results
    mapped_df = map_optimization_results(file_path)
    # Save the mapped DataFrame to a CSV file
    mapped_df.to_csv('mapped_TF_mRNA_phospho.csv', index=False)
    # Move the file to the data folder
    os.rename('mapped_TF_mRNA_phospho.csv', '../data/mapping.csv')

""" 
PAK2 doesn't exist in CollectTRI (so also not in input4.csv) but it is  
in the GeneID column of the phospho interaction file. 
No mRNA-TF estimation is possible for PAK2.  
Hence, it is not included in the output mapped file for ODE modelling.
"""
