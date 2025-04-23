# Plotting 

This module provides the `Plotter` class, which encapsulates a comprehensive suite of plotting functions designed to help visualize and analyze the results of ODE-based phosphorylation modeling. It supports a wide range of diagnostic, exploratory, and presentation plots to aid in understanding model dynamics, sensitivity, and fit quality.

## Features

- **Parallel Coordinates Plot**  
  Visualize high-dimensional ODE solution trajectories across time by plotting state variables (e.g., mRNA, protein, phosphorylated sites) along parallel axes with time as a class variable.

- **PCA Analysis**  
  - **Scree Plot (PCA Components):** Displays the individual and cumulative explained variance to help determine the number of principal components needed to capture a target amount of variance.
  - **3D PCA Plot:** Generates a three-dimensional scatter plot of the first three principal components with a smooth temporal path overlay.

- **t-SNE Plot**  
  Projects high-dimensional ODE solution data into two dimensions using t-SNE, with a temporal path drawn to highlight the progression of the system over time.

- **Parameter Bar Plot**  
  Creates bar charts summarizing the estimated kinetic parameters (e.g., phosphorylation and dephosphorylation rates) for each phosphorylation site, with colors mapped to residue positions.

- **Parameter Series Plot**  
  Plots the evolution of each estimated parameter over time (if parameter estimation is performed across multiple time points).

- **Profile Plot**  
  Visualizes adaptive profile estimation results across a defined time scale.

- **Model Fit Plot**  
  Compares the observed data with the model predictions using both matplotlib and interactive Plotly figures, enabling both static and web-based interactive visualizations.

- **Goodness-of-Fit (GoF) Plots**  
  Multiple GoF plots (gof_1 through gof_6) are provided to assess the accuracy of the model fit. These include:
  - Scatter plots comparing observed vs. fitted values with confidence intervals.
  - Plots with axis expansion and outlier annotation.
  - Cumulative distribution plots of errors.

- **Kullback-Leibler Divergence (KLD) Plot**  
  Computes and visualizes the KLD between the observed and estimated distributions, offering insight into the divergence of model predictions from the data.

- **Protein Clusters and Heatmap Plots**  
  Additional plots for exploring clustering of sensitivity values and correlations among estimated parameters.
 
--- 

## Structure

- **Private Helper:**  
  `_save_fig(fig, filename, dpi=300)` saves a matplotlib figure and then closes it.

- **Plotting Methods:**  
  - `plot_parallel`: Creates a parallel coordinates plot.
  - `pca_components`: Generates a scree plot and returns the number of principal components required.
  - `plot_pca`: Generates a 3D PCA plot.
  - `plot_tsne`: Creates a t-SNE plot with a temporal overlay.
  - `plot_param_bar`: Creates a bar chart for parameter estimates.
  - `plot_param_series`: Plots the time series evolution of estimated parameters.
  - `plot_profiles`: Plots adaptive profile estimation curves.
  - `plot_model_fit`: Compares observed data with model predictions using both matplotlib and Plotly.
  - `plot_A_S`: Produces scatter and density contour plots for A and S values.
  - Multiple `plot_gof_*` methods: Generate different goodness-of-fit plots.
  - `plot_kld`: Generates a Kullback-Leibler divergence plot.
  - `plot_clusters` and `plot_heatmap`: For visualizing protein clustering and parameter correlation.