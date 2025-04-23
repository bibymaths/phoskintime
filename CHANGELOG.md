# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-19
### Added
- Initial release of **PhosKinTime**, a Python toolkit for ODE-based modeling of phosphorylation kinetics and transcriptional time-series.
- Features include:
  - Parameter estimation.
  - Sensitivity analysis.
  - Steady-state computation.
  - Interactive visualization.
- Support for Python 3.10, 3.11, and 3.12.
- Dependencies include `numpy`, `pandas`, `seaborn`, `matplotlib`, `scipy`, and more.
- CLI entry point `phoskintime` via `bin.main:main` in root.
- Packaged directories: `bin`, `config`, `kinopt`, `models`, `paramest`, `plotting`, `sensitivity`, `steady`, `tfopt`, and `utils`.
- Documentation and homepage available at [https://bibymaths.github.io/phoskintime/](https://bibymaths.github.io/phoskintime/). 
   
## [Unreleased]
### Added
- Added light grid in plotting of model.
- Added `CHANGELOG.md`.
- Added direct link to open file from CLI.
- Added CLI wrappers for entry point.
- Added deployment configuration file.
- Added support for network via Cytoscape.
- Added configuration file for PhosKinTime settings.
- Enhanced analysis and plotting functions: added upper bound parameter, updated loss type defaults, and improved legend formatting.

### Changed
- Updated parameter bounds and model settings in configuration files.
- Refactored logging statements and improved data filtering in main processing files.

### Fixed
- Fixed display of missing kinases in output before optimization in `kinopt`.

### Removed
- Removed clipping of predicted expression.
- Deleted `abopt` directory.
