# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-19

### Added

- Initial release of **PhosKinTime**, a Python toolkit for ODE-based modeling of phosphorylation kinetics and
  transcriptional time-series.
- Features include:
    - Parameter estimation.
    - Sensitivity analysis.
    - Steady-state computation.
    - Interactive visualization.
- Support for Python 3.10, 3.11, and 3.12.
- Dependencies include `numpy`, `pandas`, `seaborn`, `matplotlib`, `scipy`, and more.
- CLI entry point `phoskintime` via `bin.main:main` in root.
- Packaged directories: `bin`, `config`, `kinopt`, `models`, `paramest`, `plotting`, `sensitivity`, `steady`, `tfopt`,
  and `utils`.
- Documentation and homepage available
  at [https://bibymaths.github.io/phoskintime/](https://bibymaths.github.io/phoskintime/).

## [0.2.0] - 2025-04-24

### Added

- Added light grid in plotting of model.
- Added `CHANGELOG.md`.
- Added direct link to open file from CLI.
- Added CLI wrappers for entry point.
- Added deployment configuration file.
- Added support for network via Cytoscape.
- Added configuration file for PhosKinTime settings.
- Enhanced analysis and plotting functions: added upper bound parameter, updated loss type defaults, and improved legend
  formatting.

### Changed

- Updated parameter bounds and model settings in configuration files.
- Refactored logging statements and improved data filtering in main processing files.

### Fixed

- Fixed display of missing kinases in output before optimization in `kinopt`.

### Removed

- Removed clipping of predicted expression.
- Deleted `abopt` directory.

## [0.3.0] - 2025-04-24

### Added

- Support for non-psite time series in kinase data.
- New results directory for structured output saving.
- Detailed docstrings and inline documentation for key functions.

### Changed

- Refactored `powell.jl`: cleaner function names, improved parameter handling, and threading support.
- Updated threading configuration and residuals calculation.
- Replaced print statements with logger output in Python modules.
- Adjusted beta bounds and default loss function settings.
- Improved ODE system equations, plotting aesthetics, and documentation structure.

### Fixed

- LaTeX formatting in README.md.
- Sheet name for estimated values in Excel export.

### Removed

- Obsolete `abopt` directory.
- Outdated module references and unused code.
- Removed julia implementation of kinopt
- Removed Project.toml for Julia dependency management.
 
## [0.4.0] - Unreleased

Added  
- Phase space plots and strip plots for state distributions in sensitivity analysis.  
- Time-state grid visualization replacing old time-wise plots.  
- Enhanced logging format for parameter bounds and model configuration.  
- Increased number of trajectories to 10,000 for improved sensitivity resolution.  
- Support for parameter relationship plots and top parameter pair visualizations.  

Changed  
- Refactored sensitivity analysis functions and configuration parameters.  
- Updated site parameter labels and adjusted development mode flags.  
- Improved aesthetics of phase space and strip plots.  
- Adjusted ODE model references (`ODE_MODEL`) and refined output normalization logic.  
- Replaced hardcoded values with computed perturbations for sensitivity analysis.  

Fixed  
- Markdown formatting in README and PYPI_README.  
- Sheet name bug in Excel export for estimated values.  

Removed  
- Deprecated analysis modes and unused constants.  
- Combined time-weight calculation from data preprocessing.
