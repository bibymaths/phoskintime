# Plotting

This module provides a comprehensive set of tools for visualizing data and results from ODE-based models of
phosphorylation dynamics. It supports various types of plots to analyze model behavior, parameter estimates, and
goodness-of-fit metrics.

## Features

- **Time-Series Visualization**:  
  Generate plots for time-dependent data, including mRNA, protein, and phosphorylation levels.

- **Dimensionality Reduction**:  
  Visualize data using PCA, t-SNE, and parallel coordinates to explore patterns and relationships.

- **Parameter Analysis**:  
  Create scatter plots, bar charts, and density plots to evaluate parameter estimates, confidence intervals, and
  sensitivity.

- **Model Fit Evaluation**:  
  Compare observed and fitted data using goodness-of-fit plots, error distributions, and Kullback-Leibler divergence.

- **Sensitivity Analysis**:  
  Visualize parameter importance and interactions using bar plots, scatter plots, radial plots, and pie charts.

- **State and Phase Space Exploration**:  
  Analyze state variability over time and phase space relationships between states.

- **Regularization and Error Metrics**:  
  Summarize regularization values and model errors across genes or experiments.

## Output

All plots are saved as high-resolution images in the specified output directory.  
Interactive visualizations (e.g., Plotly) are also supported for model fit only.