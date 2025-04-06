import pandas as pd

# Load input4.csv (which contains the Source and Target columns)
df_input4 = pd.read_csv("input4.csv")

# Load input2.csv (which should contain a column named 'GeneID')
df_input2 = pd.read_csv("input2.csv")

# Create a set of GeneID values from input2 for fast lookup
gene_ids = set(df_input2["GeneID"].dropna().astype(str))

# Filter rows in input4:
# Keep rows if either the Source or Target is found in the gene_ids set.
filtered_df = df_input4[
    df_input4["Source"].astype(str).isin(gene_ids) |
    df_input4["Target"].astype(str).isin(gene_ids)
]

# Save the filtered DataFrame to a new CSV file (or overwrite input4.csv if preferred)
filtered_df.to_csv("input4_reduced.csv", index=False)
