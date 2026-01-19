**PhosKinTime** is a modular Python package for modeling phosphorylation kinetics using ODEs. It supports multiple mechanistic hypotheses—distributive, successive, and random models—and offers tools for parameter estimation, steady-state computation, sensitivity analysis, and visualization. Designed for systems biology and phosphoproteomics research, the package enables robust, time-resolved analysis of kinase-substrate dynamics.

### Features

* Mechanistic modeling with ODEs (distributive, successive, random)
* Parameter estimation (sequential or global)
* Steady-state condition solvers
* Morris sensitivity analysis
* Publication-ready plots (fits, PCA, t-SNE, sensitivity indices)
* CLI for preprocessing, optimization, and simulation
* HTML reports for complete analysis output

### Installation

Install via pip (recommended inside a virtualenv):

```bash
pip install phoskintime
```

Or clone from GitHub for full access:

```bash
git clone https://github.com/bibymaths/phoskintime.git
cd phoskintime
pip install -r requirements.txt
```

### Usage

Run all stages of the workflow:

```bash
python phoskintime all
```

Run kinase optimization:

```bash
python phoskintime kinopt --mode evol
```

For complete usage, see the [GitHub repository](https://github.com/bibymaths/phoskintime).

### License

BSD 3-Clause License