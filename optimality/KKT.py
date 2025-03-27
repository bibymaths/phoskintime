
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the optimization results Excel file
file_path = "optimization/optimization_results.xlsx"
optimization_results = pd.ExcelFile(file_path)

# Load relevant sheets
alpha_values = pd.read_excel(file_path, sheet_name='Alpha Values')
beta_values = pd.read_excel(file_path, sheet_name='Beta Values')
residuals = pd.read_excel(file_path, sheet_name='Residuals')
estimated_values = pd.read_excel(file_path, sheet_name='Estimated Values')
observed_values = pd.read_excel(file_path, sheet_name='Observed Values')

# Validate sum constraints for Alpha values (sum of alpha for each Psite should be 1)
# This step checks the normalization constraint for \( \alpha \). Ensure the reported violations are interpreted as potential issues in parameter estimation.
alpha_sum_validation = alpha_values.groupby(['Protein', 'Psite'])['Alpha'].sum()
alpha_constraint_violations = alpha_sum_validation[alpha_sum_validation != 1]

# Validate sum constraints for Beta values (sum of beta for each Kinase should be 1)
# This step checks the normalization constraint for \( \beta \). Any violations here could indicate improper parameter bounds or fitting issues.
beta_sum_validation = beta_values.groupby(['Kinase'])['Beta'].sum()
beta_constraint_violations = beta_sum_validation[beta_sum_validation != 1]


# Plot Alpha and Beta Constraint Violations
def plot_constraint_violations(alpha_violations, beta_violations):
    alpha_violations_abs = alpha_violations.abs().groupby('Protein').sum()
    beta_violations_abs = beta_violations.abs()

    # Combine Alpha and Beta violations
    combined_violations = pd.DataFrame({
        "Alpha Violations": alpha_violations_abs,
        "Beta Violations": beta_violations_abs.reindex(alpha_violations_abs.index, fill_value=0)
    })

    # Sort by total violations
    combined_violations['Total Violations'] = combined_violations.sum(axis=1)
    combined_violations = combined_violations.sort_values(by="Total Violations", ascending=True)

    # Identify top 5 violating proteins
    top_violations = combined_violations.tail(5).index

    # Create bar plot
    plt.figure(figsize=(8, 8))
    bar_alpha = plt.bar(combined_violations.index, combined_violations["Alpha Violations"], color='dodgerblue',
                        label=r'$\alpha$')
    bar_beta = plt.bar(combined_violations.index, combined_violations["Beta Violations"],
                       bottom=combined_violations["Alpha Violations"], color='lightgreen', label=r'$\beta$')

    # Highlight top violations in red
    for bar in bar_alpha:
        if bar.get_x() in top_violations:
            bar.set_color('red')
    for bar in bar_beta:
        if bar.get_x() in top_violations:
            bar.set_color('red')

    plt.xlabel("Proteins")
    plt.ylabel("Constraint Violations")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimization/constraint_violations.png', dpi=300)
    plt.close()


# Plot Sensitivity Analysis
def plot_sensitivity_analysis(sensitivity_analysis):
    sensitivity_summary = sensitivity_analysis.groupby("GeneID")[
        ["Sensitivity Mean", "Max Sensitivity", "Min Sensitivity"]].mean()
    sensitivity_summary = sensitivity_summary.sort_values(by="Sensitivity Mean", ascending=True)

    # Create horizontal stacked bar plot
    plt.figure(figsize=(8, 8))
    bar_min = plt.barh(sensitivity_summary.index, sensitivity_summary["Min Sensitivity"], color='lightgreen',
                       label='Min')
    bar_mean = plt.barh(sensitivity_summary.index, sensitivity_summary["Sensitivity Mean"],
                        left=sensitivity_summary["Min Sensitivity"], color='dodgerblue', label='Mean')
    bar_max = plt.barh(sensitivity_summary.index, sensitivity_summary["Max Sensitivity"],
                       left=sensitivity_summary["Min Sensitivity"] + sensitivity_summary["Sensitivity Mean"],
                       color='coral', label='Max')

    plt.xlabel("Sensitivity")
    plt.ylabel("Proteins")
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimization/sensitivity.png', dpi=300)
    plt.close()


# Calculate residuals as the difference between observed and estimated values
observed_matrix = observed_values.iloc[:, 2:].values  # Exclude GeneID and Psite columns
estimated_matrix = estimated_values.iloc[:, 2:].values  # Exclude GeneID and Psite columns
residuals_matrix = observed_matrix - estimated_matrix

# Calculate gradients for primal feasibility (derivative of residuals)
# This evaluates the primal feasibility condition by analyzing residual gradients. The summary provides insights into how well the optimization aligns with observed data.
gradients = np.gradient(residuals_matrix, axis=1)
residuals_summary = {
    "Max Residual": round(np.max(residuals_matrix), 2),
    "Min Residual": round(np.min(residuals_matrix), 2),
    "Mean Residual": round(np.mean(residuals_matrix), 2),
    "Max Gradient": round(np.max(gradients), 2),
    "Min Gradient": round(np.min(gradients), 2),
    "Mean Gradient": round(np.mean(gradients), 2),
}

# Sensitivity calculation based on Eq. {eq:sensitivity}
# This computes the sensitivity of phosphorylation site contributions (\( \beta \)) to changes in kinase activity (\( P_i(t) \)). Use the sensitivity summary to identify key contributors.
Q_simulated = np.ones_like(observed_matrix)  # Assuming uniform interaction matrix
alpha_simulated = alpha_values['Alpha'].mean()  # Average alpha as a proxy for this calculation
P_observed = observed_matrix.mean(axis=1)  # Average observed phosphorylation values across time
sensitivity = Q_simulated * alpha_simulated * P_observed[:, np.newaxis]

# Summarize sensitivity
sensitivity_summary = {
    "Max Sensitivity": round(np.max(sensitivity), 2),
    "Min Sensitivity": round(np.min(sensitivity), 2),
    "Mean Sensitivity": round(np.mean(sensitivity), 2),
}


# Create LaTeX tables for residual summary and sensitivity summary
def generate_latex_table(summary_dict, table_caption):
    latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|c|}\\hline\n"
    latex_table += "Metric & Value \\\\ \hline\n"
    for key, value in summary_dict.items():
        latex_table += f"{key} & {value} \\\\ \hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{{{table_caption}}}\n\\end{table}\n"
    return latex_table


latex_residuals_table = generate_latex_table(residuals_summary, "Residual Summary")
latex_sensitivity_table = generate_latex_table(sensitivity_summary, "Sensitivity Summary")

# Print LaTeX tables
print(latex_residuals_table)
print(latex_sensitivity_table)

# Detailed sensitivity analysis
sensitivity_analysis = pd.DataFrame({
    "GeneID": observed_values.iloc[:, 0],
    "Psite": observed_values.iloc[:, 1],
    "Sensitivity Mean": P_observed,
    "Max Sensitivity": sensitivity.max(axis=1),
    "Min Sensitivity": sensitivity.min(axis=1)
})

# Identify phosphorylation sites with high sensitivity
# This identifies critical sites where changes in \( \beta \) have significant impacts. These sites are crucial for understanding and refining the model.
high_sensitivity_threshold = 0.75  # Threshold for high sensitivity
high_sensitivity_sites = np.where(sensitivity >= high_sensitivity_threshold)
genes_psites_high_sensitivity = [
    (observed_values.iloc[i, 0], observed_values.iloc[i, 1])  # GeneID, Psite
    for i in high_sensitivity_sites[0]
]

# Outputs for analysis
results = {
    "Alpha Constraint Violations": alpha_constraint_violations,
    "Beta Constraint Violations": beta_constraint_violations,
    "Residuals Summary": residuals_summary,
    "Sensitivity Summary": sensitivity_summary,
    "Detailed Sensitivity Analysis": sensitivity_analysis,
    "High Sensitivity Sites": genes_psites_high_sensitivity,
}

# Print or save results
for key, value in results.items():
    print(f"{key}:{value}\n")

# Plot violations
plot_constraint_violations(alpha_constraint_violations, beta_constraint_violations)

# Plot sensitivity analysis
plot_sensitivity_analysis(sensitivity_analysis)

# Load relevant sheets from both  files
alpha_values = pd.read_excel(file_path, sheet_name='Alpha Values')
beta_values = pd.read_excel(file_path, sheet_name='Beta Values')
residuals = pd.read_excel(file_path, sheet_name='Residuals').select_dtypes(include=['number'])

# Analyze active constraints (alpha = 0 or 1, beta = 0 or 1)
alpha_active_lower = alpha_values[alpha_values['Alpha'] == 0]
alpha_active_upper = alpha_values[alpha_values['Alpha'] == 1]
beta_active_lower = beta_values[beta_values['Beta'] == 0]
beta_active_upper = beta_values[beta_values['Beta'] == 1]

# Summarize findings
active_constraints_summary = {
    "Alpha Active Lower": len(alpha_active_lower),
    "Alpha Active Upper": len(alpha_active_upper),
    "Beta Active Lower": len(beta_active_lower),
    "Beta Active Upper": len(beta_active_upper)
}

# Merge beta values with residuals to indirectly analyze sensitivity mismatches
# Combine Beta values with Residuals
beta_with_residuals = pd.merge(
    beta_values,
    residuals.mean(axis=1).reset_index(name='Mean Residual'),
    left_index=True,
    right_on='index',
    how='left'
)

# Check mismatches for (Beta = 0 but high residuals; Beta = 1 but low residuals)
beta_lower_issues = beta_with_residuals[
    (beta_with_residuals['Beta'] == 0) & (beta_with_residuals['Mean Residual'] > 0.2)
    ]
beta_upper_issues = beta_with_residuals[
    (beta_with_residuals['Beta'] == 1) & (beta_with_residuals['Mean Residual'] < 0.1)
    ]

# Summarize sensitivity mismatch issues
sensitivity_mismatch_summary = {
    "Beta Lower Issues (High Residuals)": len(beta_lower_issues),
    "Beta Upper Issues (Low Residuals)": len(beta_upper_issues)
}

# Condition
alpha_normalization = alpha_values.groupby(['Protein', 'Psite'])['Alpha'].sum()
beta_normalization = beta_values.groupby('Kinase')['Beta'].sum()

alpha_violations = alpha_normalization[alpha_normalization != 1]
beta_violations = beta_normalization[beta_normalization != 1]

# Summarize primal feasibility results
primal_feasibility_summary = {
    "Alpha Violations": len(alpha_violations),
    "Beta Violations": len(beta_violations)
}


def print_primal_feasibility_results():
    print("Primal Feasibility Summary:")
    for key, value in primal_feasibility_summary.items():
        print(f"{key}: {value}")
    print("\nAlpha Violations:")
    print(alpha_violations)
    print("\nBeta Violations:")
    print(beta_violations)


def print_sensitivity_and_active_constraints():
    print("\nSensitivity Mismatch Summary:")
    for key, value in sensitivity_mismatch_summary.items():
        print(f"{key}: {value}")

    print("\nActive Constraints Summary:")
    for key, value in active_constraints_summary.items():
        print(f"{key}: {value}")


# Call the function to display results
print_primal_feasibility_results()

# Total number of alpha values
total_alpha = alpha_values.shape[0]
# Total number of beta values
total_beta = beta_values.shape[0]

# Print the results
print(f"Total Alpha Values: {total_alpha}")
print(f"Total Beta Values: {total_beta}")

# Equality constraints
total_alpha_normalization_constraints = alpha_values.groupby(['Protein', 'Psite']).ngroups
total_beta_normalization_constraints = beta_values['Kinase'].nunique()

# Inequality constraints
total_alpha_bounds_constraints = alpha_values.shape[0]
total_beta_bounds_constraints = beta_values.shape[0]

# Print results
print("Total Constraints:")
print(f"  Alpha Normalization: {total_alpha_normalization_constraints}")
print(f"  Beta Normalization: {total_beta_normalization_constraints}")
print(f"  Alpha Bounds: {total_alpha_bounds_constraints}")
print(f"  Beta Bounds: {total_beta_bounds_constraints}")

# Call the function to display results
print_sensitivity_and_active_constraints()