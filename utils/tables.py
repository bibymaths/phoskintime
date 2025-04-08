import os
import pandas as pd
from pandas import MultiIndex, concat

from config.constants import OUT_DIR
from kinopt.evol.config.constants import OUT_FILE
from config.logconf import setup_logger
logger = setup_logger(__name__)

def generate_tables(xlsx_file_path):
    # Load alpha and beta values from the XLSX file
    alpha_values = pd.read_excel(xlsx_file_path, sheet_name="Alpha Values")
    beta_values = pd.read_excel(xlsx_file_path, sheet_name="Beta Values")

    # Prepare the tables
    hierarchical_tables = []

    def format_float(value):
        """Custom formatter to remove trailing zeroes."""
        return f"{value:.2f}".rstrip('0').rstrip('.') if pd.notnull(value) else ""

    def merge_kinase_columns(alpha_pivot, beta_pivot):
        """Merge columns of kinases without duplicating names."""
        all_kinases = list(alpha_pivot.columns) + list(beta_pivot.columns)
        unique_kinases = sorted(set(all_kinases), key=all_kinases.index)

        merged_data = pd.DataFrame(index=alpha_pivot.index.union(beta_pivot.index), columns=unique_kinases)
        for kinase in unique_kinases:
            if kinase in alpha_pivot.columns:
                merged_data.loc[alpha_pivot.index, kinase] = alpha_pivot[kinase]
            if kinase in beta_pivot.columns:
                merged_data.loc[beta_pivot.columns, kinase] = beta_pivot[kinase].values

        return merged_data

    for protein in alpha_values['Protein'].unique():
        protein_alpha = alpha_values[alpha_values['Protein'] == protein]
        protein_beta = beta_values[beta_values['Kinase'].isin(protein_alpha['Kinase'].unique())]

        for psite in protein_alpha['Psite'].unique():
            # Filter alpha and beta data for this specific psite
            alpha_data = protein_alpha[protein_alpha['Psite'] == psite]
            beta_data = protein_beta[protein_beta['Kinase'].isin(alpha_data['Kinase'].unique())]

            # Prepare alpha and beta pivot data
            alpha_pivot = alpha_data.pivot(index='Psite', columns='Kinase', values='Alpha')
            beta_pivot = beta_data.pivot(index='Kinase', columns='Psite', values='Beta')

            # Round alpha and beta values to 3 decimal places and format
            alpha_pivot = alpha_pivot.map(format_float)
            beta_pivot = beta_pivot.map(format_float)

            # Combine alpha and beta with hierarchical levels: add latex symbol
            alpha_pivot.columns = MultiIndex.from_product([['$\\alpha$'], alpha_pivot.columns], names=['', 'Kinase'])
            beta_pivot = beta_pivot.T  # Transpose beta for matching structure
            beta_pivot.columns = MultiIndex.from_product([['$\\beta$'], beta_pivot.columns], names=['', 'Kinase'])

            # # Combine alpha and beta with hierarchical levels: DONT add latex symbol
            # alpha_pivot.columns = MultiIndex.from_product([[''], alpha_pivot.columns], names=['', 'Kinase'])
            # beta_pivot = beta_pivot.T  # Transpose beta for matching structure
            # beta_pivot.columns = MultiIndex.from_product([[''], beta_pivot.columns], names=['', 'Kinase'])

            # Concatenate alpha and beta tables
            hierarchical_table = concat([alpha_pivot, beta_pivot], axis=1)
            hierarchical_table = hierarchical_table.where(pd.notnull(hierarchical_table),
                                                          "")  # Replace NaN with empty strings
            # Rename 'Psite' to 'Site' and replace underscores with \_
            hierarchical_table.index = hierarchical_table.index.map(lambda x: str(x).replace('_', '\\_'))
            hierarchical_table.index.rename('Site', inplace=True)
            hierarchical_tables.append(((protein, psite), hierarchical_table))

    return hierarchical_tables


def save_tables(tables, output_dir):
    for (protein, psite), table in tables:
        base_filename = f"{output_dir}/{protein}_{psite.replace(':', '_')}"
        # Save as LaTeX
        with open(f"{base_filename}.tex", "w") as tex_file:
            tex_file.write(table.to_latex(multicolumn=True, multirow=True, escape=False))
        # Save as CSV
        table.to_csv(f"{base_filename}.csv")

def save_master_table(folder="latex", output_file="latex/all_tables.tex"):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".tex")])

    # Write a LaTeX file that includes all these files
    with open(output_file, "w") as out:
        out.write("% This file is auto-generated\n")
        for file in files:
            out.write(f"\\input{{{folder}/{file}}}\n")

    print(f"Generated {output_file} with {len(files)} entries.")

if __name__ == "__main__":
    xlsx_file_path = OUT_FILE
    output_dir = OUT_DIR
    # Generate hierarchical tables
    tables = generate_tables(xlsx_file_path)
    for (protein, psite), table in tables:
        logger.info(f"Protein: {protein}, Phosphorylation: {psite}")
        logger.info(table)
    # Save tables as LaTeX and CSV
    save_tables(tables, output_dir)
    save_master_table("latex", "master.tex")
    logger.info(f"LaTeX and CSV tables have been saved to {output_dir}")