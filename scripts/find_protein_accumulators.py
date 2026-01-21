import pandas as pd

# Load your picked predictions
dfp = pd.read_csv("results_global_test_proxy/pred_prot_picked.csv")
dfr = pd.read_csv("results_global_test_proxy/pred_rna_picked.csv")

# Find max values for each protein
max_p = dfp.groupby('protein')['pred_fc'].max()
max_r = dfr.groupby('protein')['pred_fc'].max()

# Calculate the coupling ratio
ratio = max_p / (max_r + 1e-6)
accumulators = ratio[ratio > 100].sort_values(ascending=False)

print("🚨 Top 'Accumulator' Proteins (Massive Protein vs Flat mRNA):")
print(accumulators.head(10))
print(f"Total accumulators found: {len(accumulators)}")