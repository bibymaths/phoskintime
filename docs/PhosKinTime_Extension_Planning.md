# 1. Drug Dose‑Response / PK‑PD / QSP Extension

## Executive summary

This extension adds a pharmacokinetics–pharmacodynamics (PK‑PD) layer to PhosKinTime so that perturbations by small‑molecule drugs can be modelled in a dose‑ and time‑dependent manner. The existing `networkmodel` integrates mRNA, total protein and phosphorylation dynamics and estimates parameters such as kinase activity multipliers and production/degradation constants[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=,factor%20scaling%20parameter). A PK‑PD extension would incorporate an exogenous drug concentration profile and link it to specific kinetic parameters or activities in the ODE system. For example, drug concentrations could modulate kinase activity multipliers (`c_k`) or degradation rates (`D_i`) through Hill/Emax functions. The extension does **not** claim to derive drug binding affinities from phosphorylation data nor replace dedicated PK modelling; it treats drug concentration as an external input that alters existing ODE terms.

## Why this is or is not close to the current framework

- **Classification:** medium‑term extension.
- **Justification:** The current framework already solves ODEs for protein/mRNA/phospho trajectories and fits parameters using JAX-based optimization[\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=predicted%20change%20should%20be%20interpreted,not%20as%20a%20standalone%20correlation). It outputs kinase activity multipliers and turnover parameters[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=,factor%20scaling%20parameter) that could be modulated by drugs. However, no PK modelling exists, and drug effects require adding new state variables (e.g., drug compartments) and dose schedules. Therefore this extension extends the core ODE system with additional compartments but remains conceptually compatible with the existing solver architecture.

## Current PhosKinTime assets this can reuse

- **Fitted kinetic parameters:** `c_k`, `A_i`, `B_i`, `C_i`, `D_i`, `Dp_i`, `E_i` and `tf_scale` from `networkmodel` outputs[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=,factor%20scaling%20parameter).
- **Predicted trajectories:** time-indexed tables for protein, mRNA and phospho states[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%60%60%60text%20TIME_POINTS_PROTEIN%20,phospho%20fitted%20trajectories).
- **Residuals and sensitivity analysis utilities:** residual outputs compare observed vs. predicted values[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=), and sensitivity analysis perturbs fitted parameters to assess influence[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=). Drug effects could reuse these analyses to quantify impact.
- **Optimization infrastructure:** the `kinopt` and `tfopt` modules provide global and local optimization frameworks with constraint handling[\[6\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/kinopt/README.md#:~:text=%2A%2Akinopt%2A%2A%20provides%20an%20end,for) and can be adapted to fit PK‑PD parameters.
- **Configuration system:** `config.toml` defines time grids and parameter bounds[\[7\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,120%2C%20240%2C%20480%2C%20960)[\[8\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,0); these can be extended to include dose schedules and PK parameters.
- **Export utilities:** existing export functions for kinase activity and phosphorylation rates[\[9\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) can be extended to include drug concentration and effect summaries.

## Required new scientific layer

A PK‑PD module must model drug absorption, distribution, metabolism and elimination as additional ODEs. At minimum, a one‑compartment model with first‑order elimination can be used:

- **PK model:**


- $$\frac{dC_{drug}}{dt} = \frac{D(t)}{V} - k_{el}C_{drug}$$
  where `D(t)` is the dosing rate, `V` is the apparent volume of distribution and `k_{el}` is the elimination rate.


- **PD link:** drug concentration modifies specific kinetic parameters through a Hill or Emax function, e.g.,


- $$c_{k}^{\text{eff}} = c_{k} \times \frac{1}{1 + \left( C_{drug}/IC_{50} \right)^{n}}$$
  for inhibition of kinase `k`, or

  $$D_{i}^{\text{eff}} = D_{i} + E_{max} \times \frac{C_{drug}}{EC_{50} + C_{drug}}$$
  for enhanced degradation.


- **Additional states:** multi‑compartment PK models (central/peripheral) or metabolite compartments could be added later.

- **Model integration:** the extended ODE system will include the drug compartment and modify the right-hand sides of existing equations via the PD link. The model remains deterministic and uses the same solver (Diffrax adaptive RK45). Parameter estimation may involve fitting `k_el`, `IC_50`, `n`, `EC_50` and `E_max` given experimental dose‑response data.

## Realistic inputs

| Filename            | Required columns                                                                                                                             | Optional columns                               | Units              | Source                                        | Notes                                                                               |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|--------------------|-----------------------------------------------|-------------------------------------------------------------------------------------|
| `dose_schedule.csv` | `time`, `dose_amount`                                                                                                                        | `route`, `duration`                            | hours, mg          | User-provided experiment or clinical protocol | Defines when and how much drug is administered. `route` may adjust bioavailability. |
| `pk_params.csv`     | `drug`, `V` (L), `k_el` (1/hr)                                                                                                               | `k_a` (absorption rate), `F` (bioavailability) | as indicated       | Literature or prior PK data                   | Baseline PK parameters; can be overridden by fitting.                               |
| `pd_targets.csv`    | `target_type` (`kinase`, `protein`, `tf`), `target_id`, `effect_type` (`inhibition`, `activation`), `IC50` or `EC50`, `n` (Hill coefficient) | `E_max`                                        | concentration (µM) | DrugBank, ChEMBL for IC50/EC50 values         | Maps the drug to specific parameters in PhosKinTime outputs.                        |
| `pkpd_measured.csv` | `time`, `concentration`                                                                                                                      | `effect_measure`                               | hours, µM          | Experimental dose‑response studies            | Used to fit PK and PD parameters.                                                   |

The extension also uses existing outputs (`kinase_activities.csv`, `predicted_protein.csv`, etc.) to map PD effects onto dynamic variables.

## Realistic input sources

- **DrugBank, ChEMBL, BindingDB:** provide IC50/EC50 and Hill coefficients for drug‑target interactions.
- **PK data repositories:** e.g., PharmGKB or published PK studies for elimination and distribution parameters.
- **LINCS/L1000 and CPTAC:** for dose‑response transcriptome and phospho‑proteome data to fit PD effects.
- **Existing PhosKinTime outputs:** predicted kinase activities and protein/phospho trajectories serve as baseline without drug intervention.

## Outputs

| Output file              | Description                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------|
| `pkpd_concentration.csv` | Simulated drug concentration versus time for each compartment.                                                |
| `pkpd_effects.csv`       | Effective parameter values (e.g., inhibited `c_k`, modified `D_i`) over time.                                 |
| `pkpd_trajectories.csv`  | Predicted protein, mRNA and phospho trajectories under drug perturbation.                                     |
| `pkpd_residuals.csv`     | Differences between measured and predicted responses in dose‑response experiments.                            |
| `pkpd_sensitivity.csv`   | Sensitivity of each PD parameter on outcome metrics (e.g., area under curve).                                 |
| `pkpd_report.json`       | Metadata: model version, input sources, PK/PD parameter estimates and uncertainty.                            |
| `pkpd_plots/`            | Visualizations of dose schedules, concentration‑time curves, dose‑response curves and overlayed trajectories. |

All tables should follow the same tidy format as existing residual and trajectory tables[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=).

## Proposed repository structure

    extensions/
      dose_response/
        __init__.py
        pk_models.py        # defines one‑ and two‑compartment PK models
        pd_links.py         # functions linking drug concentration to kinetic parameters
        simulate.py         # integrates PK and original ODE system
        fit.py              # fits PK/PD parameters to dose‑response data
        export.py           # writes outputs and plots using existing export utilities
        inputs.py           # loaders/validators for PK/PD input tables
        config.py           # default configuration schema
        cli.py              # entry point for command line execution
        plots.py            # dose‑response and PK/PD specific plots
        tests/
          test_pkpd_models.py
          test_integration.py

This `extensions` top‑level directory keeps domain‑specific code separate from the core. Shared utilities (e.g., plotting) can import from `networkmodel.export` and `protwise.plotting`.

## Configuration design

Add a `[dose_response]` block to `config.toml`:

    [dose_response]
    # List of drug names; order corresponds to pk_params and pd_targets
    name = ["drugA"]
    # Dose schedule file path (required)
    dose_schedule = "data/dose_schedule.csv"
    # PK parameter file (can be estimated)
    pk_params = "data/pk_params.csv"
    # PD target mapping file
    d_targets = "data/pd_targets.csv"
    # Measured concentration and response data for fitting (optional)
    data = "data/pkpd_measured.csv"
    # Default PK model: one_compartment, two_compartment
    model = "one_compartment"
    # Solver options for PK integration
    solver = {atol = 1e-6, rtol = 1e-5}
    # Output directory relative to run directory
    outdir = "results/pkpd/"

Comments follow the style of existing config sections[\[7\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,120%2C%20240%2C%20480%2C%20960). Additional keys (e.g., `bounds` for `IC50`) can be added later.

## CLI design

- **Python module:** `phoskintime.extensions.dose_response.cli` with click/argparse interface.
- **Command:**


- python -m phoskintime dose-response --conf config.toml


- **Pixi task:** Add to `pixi.toml`:


- [[task]]
      name = "dose-response"
      cmd = "python -m phoskintime dose-response --conf config.toml"
      category = "analysis"


- Arguments: `--fit` to estimate PK/PD parameters; `--simulate` to run simulation with fixed parameters; `--plot` to generate plots.
- Output directory: created as `<outdir>/<drug_name>/` with subfolders `trajectories`, `plots`, `metrics`, following existing result conventions.

## Minimal viable implementation

- **Inputs:** `dose_schedule.csv` with a single bolus dose; `pd_targets.csv` with one kinase target and an `IC50` value; baseline model outputs.
- **Algorithm:** One‑compartment PK with fixed `V` and `k_el`; compute drug concentration; apply a simple Emax inhibition to `c_k`; re-simulate networkmodel ODE to obtain trajectories; compute residuals vs. baseline.
- **Outputs:** Concentration curve; inhibited `c_k` values over time; perturbed trajectories; simple plots.
- **Tests:** unit tests for PK differential equation; integration test verifying that increasing dose decreases phosphorylation output; tests verifying correct loading of input tables.

## Full implementation roadmap

- **v0.1:** Add file loaders, one‑compartment PK model, Emax PD link and CLI wrapper; produce concentration and effect outputs; include minimal tests.
- **v0.2:** Support multi‑compartment PK, Hill functions and multiple targets; allow fitting `k_el`, `IC50` using measured data; integrate with `kinopt`/`tfopt` optimization to co‑fit PD parameters.
- **v0.3:** Add plotting utilities (dose‑response curves, sensitivity analysis) and export to dashboard; implement configuration validation.
- **v0.4:** Include advanced QSP features such as metabolite compartments or saturable elimination; integrate with `BayesianInference` to estimate posterior distributions; provide example notebooks.
- **v1.0:** Stable release with documentation, reproducible examples and cross‑validation on published drug perturbation datasets.

## Validation strategy

- **Unit tests:** verify PK ODE solutions against analytical solutions for bolus dosing; check PD scaling functions produce expected limiting values.
- **Integration tests:** run simulation with and without drug and ensure the model output changes appropriately (e.g., decreased phosphorylation when inhibiting kinases).
- **Synthetic data tests:** generate synthetic dose‑response curves using known PK/PD parameters and ensure the fitter recovers them within tolerance.
- **Regression tests:** fix a seed and verify that changes in implementation do not alter existing results.
- **Biological sanity checks:** compare predicted IC50 and elimination half‑lives with literature values; check that predicted changes in kinase activity multipliers are plausible relative to measured potency.
- **External benchmarks:** where possible, compare simulated responses to LINCS/CPTAC dose‑response experiments for the same drugs.

## Risks and failure modes

- **Parameter non‑identifiability:** PK and PD parameters may be poorly identifiable from limited dose‑response data, leading to wide uncertainty.
- **Scale mismatch:** integrating PK models can introduce timescales (minutes/hours) different from cellular phosphorylation timescales (minutes); solver stiffness may increase.
- **Insufficient data:** high‑quality dose‑response phosphoproteomics is scarce; the extension may rely on assumptions about potency.
- **Overfitting:** fitting PK/PD parameters concurrently with network parameters may overfit small datasets.
- **Improper mapping:** incorrect mapping of drug targets to model parameters could misrepresent mechanism of action.

## What not to do

- Do **not** infer drug binding affinities from phosphorylation data alone.
- Do **not** claim that the PK model is physiologically accurate beyond the simple compartment(s) defined.
- Do **not** hard‑code drug‑specific values; keep parameters configurable.
- Do **not** alter the core `networkmodel` equations; extend them through a wrapper.
- Do **not** ignore parameter bounds defined in existing configuration files[\[8\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,0).

## Recommended priority

**Second**. Extending PhosKinTime with a PK‑PD layer is technically feasible and directly relevant for pharmacological studies. The required ODE and optimization infrastructure aligns with the existing framework, but the development effort is moderate due to new compartments and external data dependencies.

# 2. mRNA Vaccine / Antigen‑Expression / Immune‑Response Extension

## Executive summary

This extension models how synthetic mRNA vaccines or gene therapy constructs drive antigen expression and trigger innate immune responses. PhosKinTime currently simulates endogenous mRNA, protein and phosphorylation dynamics[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%60%60%60text%20TIME_POINTS_PROTEIN%20,phospho%20fitted%20trajectories). The vaccine extension would add an exogenous mRNA species encoding an antigen; its translation would produce antigenic protein, which could be processed and presented to immune effectors. The framework would then simulate downstream signalling pathways (e.g., interferon responses) by introducing additional state variables for cytokines and immune modulators. The extension does **not** claim to model adaptive immunity (e.g., T cell clonal expansion) or to predict vaccine efficacy; it focuses on early antigen expression and innate signalling in the same timescale as phospho‑proteomics experiments.

## Why this is or is not close to the current framework

- **Classification:** downstream annotation layer / medium-term extension.
- **Justification:** The current ODE system describes gene expression and post‑translational modifications within cells. Adding an exogenous mRNA vaccine is conceptually similar to adding a new mRNA/protein species but introduces novel processes: exogenous mRNA uptake, translation, antigen processing and innate immune sensing. A minimal model could reuse existing translation and degradation parameters but needs new states for antigen and cytokine species and may require coupling to pattern‑recognition receptor pathways. Therefore it sits at a medium distance from the core but remains within dynamic modelling.

## Current PhosKinTime assets this can reuse

- **Translation kinetics:** existing parameters for mRNA degradation (`B_i`) and protein production (`C_i`) and turnover (`D_i`)[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=,factor%20scaling%20parameter) can be reused to simulate antigen translation and degradation.
- **Predicted trajectories and residual analyses:** existing pipeline writes mRNA and protein fitted trajectories[\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%60%60%60text%20TIME_POINTS_PROTEIN%20,phospho%20fitted%20trajectories) and residuals[\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=). These utilities can be reused to visualise antigen expression and immune response curves.
- **Sensitivity analysis infrastructure:** to quantify which kinetic parameters (e.g., vaccine dose, translation rate) most influence antigen peak concentration[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=).
- **Dashboard and export functions:** can be extended to include antigen and cytokine trajectories in interactive dashboards.
- **Configuration and CLI patterns:** reuse the `config.toml` schema and CLI skeleton for new tasks[\[7\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,120%2C%20240%2C%20480%2C%20960).

## Required new scientific layer

- **Exogenous mRNA uptake and decay:** a new state variable `M_v(t)` representing vaccine mRNA copies with uptake kinetics and degradation rate `k_dec_v`.

$$\frac{dM_{v}}{dt} = u_{v}(t) - k_{dec\_ v}M_{v}$$

where `u_v(t)` is the dosing schedule.

- **Antigen translation and degradation:** a new protein state `P_v(t)` produced from `M_v` with translation rate `k_trans_v` and degraded at `k_deg_v`.

- **Innate immune sensing:** simplified pattern‑recognition receptor activation triggered by `M_v` or `P_v` could be modelled as a sigmoidal function producing an interferon or cytokine state `I(t)`.

$$\frac{dI}{dt} = k_{I}\frac{M_{v}^{h}}{K_{I}^{h} + M_{v}^{h}} - k_{clear}I$$

- **Impact on network:** cytokine signalling could modulate transcription factors (`tf_scale`), basal production (`A_i`) or degradation parameters (`B_i`,`C_i`) for select genes. A mapping file would specify which PhosKinTime parameters are impacted.

- **Optional modules:** translation efficiency modifiers for codon optimisation, mRNA secondary structure stability, or lipid nanoparticle delivery could be incorporated in later versions.

## Realistic inputs

| Filename                 | Required columns                                                | Optional columns       | Units              | Source                                                       | Notes                                                                         |
|--------------------------|-----------------------------------------------------------------|------------------------|--------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------|
| `vaccine_schedule.csv`   | `time`, `dose_mRNA`                                             | `formulation`, `route` | hours, µg          | User‑provided                                                | Specifies injection time and dose; `formulation` may affect uptake rate.      |
| `vaccine_params.csv`     | `k_uptake`, `k_dec_v`, `k_trans_v`, `k_deg_v`                   | `codon_opt_score`      | rate constants     | Literature on mRNA vaccine kinetics; pre‑clinical PK/PD data | Parameter guesses or priors for vaccine mRNA translation and degradation.     |
| `immune_params.csv`      | `k_I`, `K_I`, `h`, `k_clear`                                    | `target_mapping`       | as indicated       | Immunology literature                                        | Defines cytokine dynamics.                                                    |
| `antigen_annotation.csv` | `gene`, `antigen_peptide`, `MHC_allele`, `immunogenicity_score` |                        | sequences/affinity | IEDB, ImmPort                                                | Connects antigen translation products to known epitopes.                      |
| `immune_targets.csv`     | `target_type`, `target_id`, `effect`, `magnitude`               |                        | dimensionless      | Expert‑curated                                               | Maps cytokine levels to modifications of PhosKinTime parameters.              |
| `vaccination_data.csv`   | `time`, `measured_antigen`, `measured_cytokine`                 |                        | hours              | Experimental data                                            | For fitting vaccine kinetic parameters and validating immune response curves. |

## Realistic input sources

- **Immune Epitope Database (IEDB):** epitope sequences, MHC binding affinities and immunogenicity scores.
- **ImmPort and Expression Atlas:** innate immune response time courses after mRNA vaccination.
- **GEO / ArrayExpress:** transcriptomic and proteomic datasets post‑vaccination for parameter fitting.
- **Vaccine PK studies:** published data on mRNA stability and translation kinetics.

## Outputs

| Output file                    | Description                                                                                 |
|--------------------------------|---------------------------------------------------------------------------------------------|
| `vaccine_antigen.csv`          | Simulated antigen mRNA and protein levels vs. time.                                         |
| `vaccine_immune.csv`           | Innate immune mediator (e.g., interferon) levels vs. time.                                  |
| `vaccine_effects.csv`          | Effective modifications to PhosKinTime parameters (e.g., `tf_scale` modulation).            |
| `vaccination_trajectories.csv` | Predicted trajectories of endogenous mRNA, protein and phospho species under vaccination.   |
| `vaccine_residuals.csv`        | Differences between measured and predicted antigen and cytokine levels.                     |
| `vaccine_plots/`               | Plots of antigen expression and immune response; overlay with experimental data.            |
| `vaccine_report.json`          | Metadata summarizing vaccine kinetics parameters, immunogenicity annotation and provenance. |

## Proposed repository structure

    extensions/
      vaccine/
        __init__.py
        mRNA.py             # vaccine mRNA uptake and decay models
        antigen.py          # translation and degradation of antigen protein
        immune.py           # innate immune sensing and cytokine dynamics
        coupling.py         # maps immune mediators to PhosKinTime parameters
        simulate.py         # integrates vaccine and network ODEs
        fit.py              # fits kinetic parameters using experimental data
        export.py           # writes outputs and generates plots
        inputs.py           # validators for vaccine-related tables
        config.py           # configuration schema
        cli.py              # command-line interface
        tests/
          test_vaccine_models.py
          test_coupling.py

The folder lives under `extensions` to avoid polluting core modules. `simulate.py` will import the existing `networkmodel` solver and apply parameter modifications.

## Configuration design

Add a `[vaccine]` section to `config.toml`:

    [vaccine]
    # Path to dosing schedule
    dose_schedule = "data/vaccine_schedule.csv"
    # Parameter files
    vaccine_params = "data/vaccine_params.csv"
    immune_params = "data/immune_params.csv"
    antigen_annotation = "data/antigen_annotation.csv"
    immune_targets = "data/immune_targets.csv"
    # Output directory
    outdir = "results/vaccine/"
    # Flags
    fit = true             # fit kinetic parameters if data present
    use_codons = false     # apply codon optimisation modifiers

Each parameter file contains columns described above. Comments and keys follow existing style[\[7\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,120%2C%20240%2C%20480%2C%20960).

## CLI design

- **Python module:** `phoskintime.extensions.vaccine.cli`.
- **Command:**


- python -m phoskintime vaccine --conf config.toml [--fit] [--simulate]


- **Pixi task:** add `vaccine` under `[[task]]` with the same command.
- Optional flags: `--fit` fits kinetic parameters, `--simulate` runs simulation with given parameters, `--plot` draws antigen and immune curves.
- Output directories follow the pattern `<outdir>/<dose_schedule_file>/` with subdirectories for trajectories, plots and metrics.

## Minimal viable implementation

- **Inputs:** A single vaccine dose schedule; default parameter values for uptake, translation and immune sensing; mapping file linking interferon to a global transcription factor scaling parameter.
- **Algorithm:** Solve additional ODEs for `M_v`, `P_v` and `I` using explicit Euler or Diffrax integrator; update `tf_scale` or `A_i` in `networkmodel` with a linear function of `I`; run simulation; produce antigen and immune curves and overlay with any measured data.
- **Outputs:** Antigen and immune trajectories; predicted modifications to gene expression; simple plots.
- **Tests:** unit tests ensuring the antigen ODE yields expected exponential decay; integration test verifying that increasing vaccine dose increases antigen levels; test that immune coupling modifies network parameters.

## Full implementation roadmap

- **v0.1:** Implement basic mRNA and antigen ODEs and coupling to `tf_scale`; produce antigen and immune plots; write CLI and configuration; include tests.
- **v0.2:** Include innate immune sensing via Hill kinetics; allow parameter fitting to measured cytokine data; support multiple antigens; integrate codon optimisation modifiers.
- **v0.3:** Add mapping between antigen sequences and immunogenicity scores using IEDB; export epitope annotation; include interactive dashboard elements.
- **v0.4:** Extend to poly‑epitope constructs; include simulation of memory cytokines; provide example notebooks with vaccination time courses.
- **v1.0:** Publish validated extension with real vaccine datasets and cross‑validated predictions.

## Validation strategy

- **Unit tests:** check ODE solutions and coupling functions; ensure codon optimisation multiplier behaves correctly.
- **Integration tests:** simulate a simple vaccination scenario and verify that antigen and immune curves follow expected kinetics; ensure coupling modifies gene expression in network outputs.
- **Synthetic data tests:** create synthetic vaccine and immune time courses; test that fitting recovers the known parameters.
- **Biological sanity checks:** compare simulated antigen half‑life and interferon dynamics with published data; ensure predicted modulation of transcription factor scaling falls within realistic bounds.
- **External benchmarks:** if datasets from vaccine studies are available, fit the model and compare predicted cytokine profiles and gene expression changes.

## Risks and failure modes

- **Complexity of immune responses:** innate immune signalling involves many pathways; the simplified model may capture only a subset of dynamics.
- **Data scarcity:** high‑quality time‑series data for antigen translation and cytokine levels are limited; parameter estimates may be uncertain.
- **Coupling assumptions:** mapping interferon to transcription factor scaling may be simplistic; real regulatory effects are gene‑specific.
- **Parameter identifiability:** multiple parameters (uptake, degradation, translation) might yield similar antigen trajectories.
- **Numerical stiffness:** addition of rapid immune reactions could stiffen the ODE system and require careful solver settings.

## What not to do

- Do **not** attempt to model adaptive immunity or antibody affinity maturation; this extension focuses on early cellular responses.
- Do **not** claim to predict vaccine efficacy or protective immunity from phosphorylation data.
- Do **not** incorporate proprietary vaccine formulation details; only publicly available parameters should be included.
- Do **not** modify core `networkmodel` files; implement coupling through wrappers.
- Do **not** ignore mRNA translation and degradation parameters when simulating antigen production.

## Recommended priority

**Later.** While scientifically interesting, this extension introduces new biological domains (vaccine delivery and innate immunity) and requires external data and assumptions. It should be implemented after a PK‑PD extension and structural annotations are in place.

# 3. Antibody Perturbation / Antibody‑Response Extension

## Executive summary

This extension models how exogenous antibodies bind to cellular proteins or phospho‑epitopes and perturb signalling dynamics. PhosKinTime currently predicts phosphorylation rates and kinase activities[\[9\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=), but does not represent antibody binding or immune responses. The antibody extension would introduce binding kinetics (association and dissociation) between antibodies and their targets, reducing the availability or activity of the bound proteins. It can simulate neutralisation (removal of active protein) or crosslinking (aggregation). The extension does **not** perform antibody design; it uses user‑supplied binding parameters and epitope mapping.

## Why this is or is not close to the current framework

- **Classification:** downstream annotation layer / external pipeline integration.
- **Justification:** PhosKinTime focuses on intracellular kinetics; antibody perturbation acts at the extracellular or cell‑surface level and may not be captured by existing parameters. However, some antibodies (e.g., phospho‑specific antibodies) directly sequester phospho‑sites and can be represented as an additional inhibitory term on phosphorylation rates. Thus the extension can leverage existing phosphorylation dynamics but requires new state variables for antibody‑antigen complexes and new input mapping. It remains outside the core but can interface through targeted parameter modifications.

## Current PhosKinTime assets this can reuse

- **Phosphorylation rates and kinase activity outputs:** predicted phosphorylation rates per site[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation) and kinase activities[\[9\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) can be targeted by antibodies.
- **Residue-level mapping:** data preprocessing produces mappings of kinase‑phosphorylation interactions and gene identifiers[\[11\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md#:~:text=,mapping%20files%20for%20further%20analysis); this can be extended to map antibody targets.
- **Sensitivity analysis:** can identify phospho‑sites or proteins whose perturbation significantly alters network behaviour[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=), guiding antibody targeting strategies.
- **Optimization and simulation engine:** the same ODE solver can incorporate additional binding ODEs; the export and dashboard modules can display perturbed trajectories.

## Required new scientific layer

- **Binding kinetics:** For each antibody/epitope pair, introduce variables for free antibody (`Ab`), free antigen (`Ag`) and bound complex (`AbAg`). Mass‑action kinetics:

$$\frac{dAb}{dt} = - k_{on}Ab \times Ag + k_{off}AbAg - k_{clear}Ab$$

$$\frac{dAg}{dt} = - k_{on}Ab \times Ag + k_{off}AbAg + ...$$

$$\frac{dAbAg}{dt} = k_{on}Ab \times Ag - k_{off}AbAg - k_{clear\_ complex}AbAg$$

`Ag` corresponds to the concentration of a specific protein or phospho‑site predicted by PhosKinTime; `Ab` is the antibody concentration input; `k_on`, `k_off`, `k_clear` are binding and clearance rates.

- **Effect on signalling:** Bound antigen is considered inactive. For kinase targets, effective kinase activity multiplier is reduced; for phospho‑site targets, forward phosphorylation rate scaling `kappa_{g,s}` could be modulated.

- **Epitope mapping:** A mapping file linking antibody names to gene symbols, phosphorylation site positions or domains (e.g., Y1234) is required. For antibodies targeting conformational epitopes, integration with the structural extension (Section 4) may be necessary.

- **Antibody clearance:** Antibody pharmacokinetics may be modelled similarly to the PK extension (Section 1) but often with longer half‑life; optional compartments can be added.

## Realistic inputs

| Filename              | Required columns                                                                  | Optional columns        | Units           | Source                        | Notes                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------|-------------------------|-----------------|-------------------------------|-----------------------------------------------------------------------------------------------------------|
| `antibody_dose.csv`   | `time`, `dose_concentration`                                                      | `isotype`               | hours, µM       | User-provided dosing protocol | Defines when antibodies are administered and at what concentration.                                       |
| `antibody_params.csv` | `antibody`, `target_type`, `target_id`, `k_on`, `k_off`, `k_clear`, `effect_type` | `Kd`, `stoichiometry`   | 1/(µM hr), hr⁻¹ | Literature, BindingDB         | Binding kinetics per antibody-target pair; `effect_type` describes how binding modifies model parameters. |
| `epitope_mapping.csv` | `antibody`, `gene`, `psite_position`, `residue`, `affinity_score`                 | `structure_id`          |                 | IEDB, PDB                     | Maps antibodies to specific proteins or phospho‑sites; may include structural annotations.                |
| `antibody_pk.csv`     | `k_abs`, `k_el`, `volume`                                                         |                         | hr⁻¹, hr⁻¹, L   | Pharmacokinetic studies       | Optional PK model for antibody clearance.                                                                 |
| `antibody_data.csv`   | `time`, `antigen_activity`                                                        | `complex_concentration` | hours           | Experimental data             | For fitting binding parameters and validating model predictions.                                          |

## Realistic input sources

- **IEDB and ImmPort:** epitope and monoclonal antibody binding data.
- **BindingDB and DrugBank:** affinity constants (`Kd`), association/dissociation rates for antibody‑protein pairs.
- **Therapeutic antibody databases (e.g., Thera‑SAbDab):** sequences, isotypes and kinetic parameters.
- **Bioinformatics tools:** predicted epitopes using Bepipred, DiscoTope; structural mapping via PDB or AlphaFold; cross‑reference with `processing/map.py` outputs[\[11\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md#:~:text=,mapping%20files%20for%20further%20analysis).

## Outputs

| Output file                 | Description                                                                                |
|-----------------------------|--------------------------------------------------------------------------------------------|
| `antibody_binding.csv`      | Simulated free and bound antibody/antigen concentrations over time.                        |
| `antibody_effects.csv`      | Effective modifications to kinase activities or phosphorylation rate constants over time.  |
| `antibody_trajectories.csv` | Predicted mRNA, protein and phospho trajectories under antibody perturbation.              |
| `antibody_residuals.csv`    | Differences between measured and predicted antigen activities and phosphorylation signals. |
| `antibody_sensitivity.csv`  | Sensitivity of binding parameters (e.g., `k_on`, `k_off`) on signalling outcomes.          |
| `antibody_report.json`      | Metadata summarizing binding parameters, epitope mapping, and provenance.                  |
| `antibody_plots/`           | Plots of antibody concentration, binding kinetics, and perturbed trajectories.             |

## Proposed repository structure

    extensions/
      antibody/
        __init__.py
        binding.py       # defines mass‑action binding kinetics and antibody PK
        coupling.py      # maps bound complexes to PhosKinTime parameter modifications
        simulate.py      # integrates binding ODEs with network model
        fit.py           # fits binding parameters to experimental data
        export.py        # writes outputs and plots
        inputs.py        # validates antibody-related tables
        config.py        # configuration schema
        cli.py
        tests/
          test_binding.py
          test_coupling.py

## Configuration design

Add `[antibody]` to `config.toml`:

    [antibody]
    dose_schedule = "data/antibody_dose.csv"
    params = "data/antibody_params.csv"
    epitopes = "data/epitope_mapping.csv"
    ab_pk = "data/antibody_pk.csv"  # optional
    fit = false
    outdir = "results/antibody/"

This minimal section points to the input files; optional flags can enable parameter fitting or PK modelling.

## CLI design

- **Python module:** `phoskintime.extensions.antibody.cli`.
- **Command:**


- python -m phoskintime antibody --conf config.toml [--fit] [--simulate]


- **Pixi task:** add an `antibody` task with the same command.
- Flags: `--fit` to estimate `k_on`, `k_off` and clearance; `--simulate` to run with fixed parameters; `--plot` to generate binding and trajectory plots.
- Output directories: `<outdir>/<antibody_name>/` with subfolders for binding, trajectories and plots.

## Minimal viable implementation

- **Inputs:** Single antibody targeting one phospho‑site; known `k_on` and `k_off`; a simple dose schedule.
- **Algorithm:** Simulate binding kinetics using ordinary differential equations; compute effective reduction in phosphorylation rate scaling; run `networkmodel` with modified `kappa_{g,s}`; output perturbed trajectories.
- **Outputs:** Antibody binding curve; reduction in target phosphorylation; predicted time courses of protein/phospho levels; simple plots.
- **Tests:** unit test for binding ODE; integration test verifying that increased antibody concentration reduces phosphorylation output; test that mapping between epitope and parameter works.

## Full implementation roadmap

- **v0.1:** Implement binding kinetics and simple inhibitory coupling; create CLI and configuration; provide tests.
- **v0.2:** Support multiple antibodies and multiple targets; incorporate stoichiometry and crosslinking effects; optional PK model for antibody clearance.
- **v0.3:** Enable fitting of binding parameters using measured binding or functional data; integrate with sensitivity analysis to prioritise targets.
- **v0.4:** Incorporate structural epitope mapping from Section 4; include antibody isotype effects (e.g., Fc‑mediated clearance); add example notebooks.
- **v1.0:** Provide a validated extension with documentation and tested on therapeutic antibody perturbation datasets.

## Validation strategy

- **Unit tests:** check mass‑action binding solutions; ensure that the mapping of bound complex reduces the corresponding kinetic parameter.
- **Integration tests:** simulate a scenario with known binding parameters and verify expected reduction in phosphorylation; ensure that adding more antibodies intensifies inhibition.
- **Synthetic data tests:** generate synthetic binding and signalling data; fit `k_on`, `k_off` and verify recovery.
- **Biological sanity checks:** compare predicted binding half‑life and occupancy with literature; ensure that the model does not predict antibody levels beyond plausible physiological range.
- **External benchmarks:** use published antibody perturbation proteomics datasets to validate predicted effects on phosphorylation and downstream transcription.

## Risks and failure modes

- **Mapping uncertainty:** epitope mapping may be uncertain; mismapping an antibody to the wrong site could mislead predictions.
- **Data limitations:** kinetic constants are often unavailable for specific antibodies and conditions; may require assumptions.
- **In vivo complexity:** antibody effects include immune effector functions (e.g., ADCC) not modelled here; ignoring them may oversimplify biological outcomes.
- **Non‑identifiability:** `k_on` and `k_off` may be correlated, leading to poorly constrained fits.
- **Numerical stiffness:** adding binding kinetics may increase stiffness, requiring careful solver settings.

## What not to do

- Do **not** claim to design antibodies or predict epitope sequences; rely on external epitope mapping and kinetics.
- Do **not** assume that antibody binding always results in complete inhibition; allow partial occupancy.
- Do **not** ignore clearance of antibodies or complexes when modelling long time courses.
- Do **not** modify core ODEs; implement binding through additional compartments.
- Do **not** embed proprietary antibody sequences.

## Recommended priority

**Later / downstream integration.** Although antibody perturbations are biologically relevant, the complexity of binding kinetics and scarcity of high‑quality time‑course data make this a lower priority. It should be developed after PK‑PD and structural annotation extensions are established.

# 4. Structural Bioinformatics Extension

## Executive summary

This extension provides structural annotations for proteins and phosphorylation sites represented in PhosKinTime. It does not alter the dynamic model but enriches outputs with domain, secondary structure, solvent accessibility and homology information. These annotations can aid interpretation of fitted parameters and prioritise targets for experimental validation or MD simulations. The extension maps each protein (gene) to a UniProt accession, retrieves or predicts a structure (PDB or AlphaFold), and uses tools such as DSSP or FreeSASA to compute per‑residue features. It also maps phosphorylation sites to 3D coordinates and determines whether they lie in ordered domains, disordered regions or interfaces. The extension does **not** perform structural modelling or design; it annotates existing structures.

## Why this is or is not close to the current framework

- **Classification:** downstream annotation layer.
- **Justification:** The current model does not use structural information; outputs are purely temporal and quantitative. Structural annotation does not change the ODE system but adds metadata and derived metrics. It can be implemented as a separate post‑processing step that reads existing outputs (phosphorylation rates, sensitivities, mapping tables) and enriches them. Therefore it is conceptually decoupled and safe to implement without altering core models.

## Current PhosKinTime assets this can reuse

- **Mapped gene and phosphorylation site identifiers:** The preprocessing mapping step produces tables `mapping.csv` and `nodes.csv` linking kinases, genes and phospho‑sites[\[12\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md#:~:text=,).
- **Phosphorylation rate outputs:** `export_S_rates` summarises phosphorylation rates by protein and site[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation). These can be annotated with structural features.
- **Sensitivity rankings:** sensitivity analysis identifies influential parameters or sites[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=); structural annotation can prioritise high‑impact sites.
- **Configuration system:** a new section can be added without affecting existing modules. Export utilities and dashboard can be extended to include structural columns.

## Required new scientific layer

- **Sequence-to-structure mapping:** map each gene symbol to a UniProt accession using external databases (e.g., MyGeneInfo) and retrieve structures:

- If a resolved structure exists in the Protein Data Bank, download the PDB file.

- Otherwise, retrieve an AlphaFold model from AlphaFold DB or EBI PDB. Use the highest‑confidence model for the species of interest.

- **Phosphosite mapping:** use the residue number and amino acid from the phosphorylation site (from `psite_position` and `residue` fields) to locate the corresponding residue in the structure. Consider isoform numbering; if mapping fails, flag the site for manual curation.

- **Feature computation:** compute per‑residue annotations:

- Secondary structure (helix, sheet, loop) using DSSP.

- Solvent accessible surface area (SASA) using FreeSASA or DSSP output.

- Domain and motif annotations via Pfam/InterPro.

- Disorder prediction using tools like IUPred or by absence of structure.

- **Integrative summary:** join computed features with phosphorylation rate and sensitivity tables. Provide summary statistics (e.g., average SASA for high‑sensitivity sites). Optionally export PyMOL session scripts to visualise highlighted sites.

## Realistic inputs

| Filename                | Required columns                                                             | Optional columns | Units | Source                      | Notes                                                                 |
|-------------------------|------------------------------------------------------------------------------|------------------|-------|-----------------------------|-----------------------------------------------------------------------|
| `structure_sources.csv` | `gene`, `uniprot_id`, `preferred_pdb_id`                                     | `isoform`        |       | UniProt, MyGeneInfo         | Map gene to structure; `preferred_pdb_id` can be blank for AlphaFold. |
| `psite_mapping.csv`     | `gene`, `residue`, `position`, `structure_chain`, `structure_residue_number` |                  |       | Generated by mapping script | Connects phospho‑sites to structure residues.                         |
| `features_config.csv`   | `feature`, `tool`, `parameters`                                              |                  |       | User configuration          | Lists which structural features to compute and tool settings.         |
| `structures/`           | downloaded PDB or mmCIF files                                                |                  |       | RCSB PDB, AlphaFold DB      | Local storage for structures.                                         |

## Realistic input sources

- **UniProt and MyGeneInfo:** mapping from gene symbol to UniProt accession and isoform information.
- **Protein Data Bank (PDB):** resolved protein structures.
- **AlphaFold DB:** predicted structures when no PDB exists.
- **Pfam / InterPro:** domain and motif annotations.
- **DSSP / FreeSASA:** tools for secondary structure and SASA computation.
- **Disorder prediction servers:** IUPred, MobiDB-lite for intrinsic disorder.

## Outputs

| Output file                 | Description                                                                                                                                                                                                                                                            |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `structure_annotations.csv` | Per‑residue table with secondary structure, SASA, domain and disorder information.                                                                                                                                                                                     |
| `psite_structural.csv`      | Phosphorylation sites annotated with structural features and mapped coordinates.                                                                                                                                                                                       |
| `site_prioritization.csv`   | Ranked list of phospho‑sites combining sensitivity score, phosphorylation rate[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation) and structural features (e.g., surface exposure). |
| `protein_models/`           | Downloaded PDB/mmCIF or AlphaFold structures, renumbered and trimmed to match sequences.                                                                                                                                                                               |
| `pymol_sessions/`           | PyMOL or ChimeraX session files highlighting high‑impact sites.                                                                                                                                                                                                        |
| `structural_report.json`    | Metadata summarizing methods, data sources and versioning.                                                                                                                                                                                                             |

## Proposed repository structure

    extensions/
      structural/
        __init__.py
        fetch_structures.py    # download PDB/AlphaFold models given UniProt IDs
        map_sites.py           # map phospho positions to structure residue numbers
        compute_features.py    # run DSSP, FreeSASA and disorder predictors
        prioritize.py          # rank sites combining structural and dynamic metrics
        export.py              # write annotation tables and PyMOL scripts
        config.py              # configuration schema
        cli.py
        tests/
          test_mapping.py
          test_feature_computation.py

The structural extension is separated from dynamic modelling; it reads existing outputs and writes annotation files.

## Configuration design

Add `[structural]` to `config.toml`:

    [structural]
    # Path to mapping of genes to UniProt accessions
    source_map = "data/structure_sources.csv"
    # Directory to store downloaded structures
    structure_dir = "data/structures/"
    # Feature configuration file
    features = "data/features_config.csv"
    # Whether to compute disorder predictions
    compute_disorder = true
    # Output directory
    outdir = "results/structural/"

## CLI design

- **Python module:** `phoskintime.extensions.structural.cli`.
- **Command:**


- python -m phoskintime structural --conf config.toml [--fetch] [--annotate] [--prioritize]


- **Pixi task:** add `structural` command with the same options.
- Flags: `--fetch` downloads structures; `--annotate` maps sites and computes features; `--prioritize` ranks sites; `--plot` could generate structural summaries.
- Output directories follow `<outdir>/<gene>/` for gene‑specific annotations.

## Minimal viable implementation

- **Inputs:** Mapping file for a small set of genes; list of phospho‑sites; simple feature configuration (secondary structure and SASA only).
- **Algorithm:** Fetch structures from PDB/AlphaFold; map phospho‑site positions to residue numbers; run DSSP to compute secondary structure and SASA; join with phosphorylation rate table; export annotated table.
- **Outputs:** `psite_structural.csv` with columns for residue, secondary structure and SASA; simple ranking combining high SASA and high phosphorylation rate.
- **Tests:** unit tests for mapping (correct residue numbers), feature computation (expected output length) and prioritisation (ties broken consistently).

## Full implementation roadmap

- **v0.1:** Implement structure fetching and site mapping; compute secondary structure and SASA; basic ranking; include tests.
- **v0.2:** Add disorder prediction and domain/motif annotations from Pfam/InterPro; support isoforms; compute solvent exposure categories (buried vs. exposed).
- **v0.3:** Integrate sensitivity scores[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) and phosphorylation rate rankings[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation); export combined ranking; generate PyMOL sessions; include interactive dashboards.
- **v0.4:** Provide precomputed annotations for common organisms; support alternative structural sources (e.g., Swiss‑Model); implement caching and update mechanisms.
- **v1.0:** Stable release with documentation, validated mapping on test datasets and reproducible examples.

## Validation strategy

- **Unit tests:** verify that residue mapping correctly handles numbering offsets and missing residues; check that feature computation tools are invoked and outputs match expected dimensions.
- **Integration tests:** run the full pipeline on a test dataset (e.g., a small set of proteins) and check that outputs contain expected columns and values; verify that high‑SASA sites are correctly identified.
- **Cross‑validation:** compare computed secondary structure and SASA with annotations from existing resources (e.g., DSSP or PDBsum); confirm mapping accuracy using manual inspection for a subset.
- **Regression tests:** ensure that adding new features or updating databases does not break existing functionality.

## Risks and failure modes

- **Mapping errors:** mismatches between gene isoforms, sequence numbering and structure numbering can lead to incorrect site annotations.
- **Data availability:** some proteins may lack resolved structures; AlphaFold predictions may have low confidence in certain regions.
- **Tool dependencies:** DSSP, FreeSASA and disorder predictors must be installed; version differences can yield different results.
- **Computational cost:** large numbers of proteins/sites could slow computation; caching and parallelisation may be needed.
- **Overinterpretation:** structural annotation does not confirm functional importance; high SASA does not guarantee accessibility to kinases or antibodies.

## What not to do

- Do **not** attempt de novo structure prediction or MD simulations; rely on existing structures or AlphaFold models.
- Do **not** override core outputs; structural annotation is supplementary.
- Do **not** claim that structural features alone determine phosphorylation or druggability; use them in combination with dynamic metrics.
- Do **not** embed large structural files in the repository; download and cache them externally.
- Do **not** mix isoforms without explicit mapping.

## Recommended priority

**First.** Structural annotation is independent of dynamic modelling and provides immediate value by contextualising fitted parameters. It requires no changes to core ODE models and can be implemented as a post‑processing pipeline using publicly available tools and databases.

# 5. Molecular Dynamics Simulation Prioritization Extension

## Executive summary

This extension prioritises proteins or phosphorylation sites for detailed molecular dynamics (MD) simulations based on PhosKinTime outputs and structural annotations. It does **not** perform MD simulations itself; rather, it generates ranking tables, prepares input files and optionally runs lightweight energy minimisations to check structural feasibility. The goal is to select a manageable subset of sites or protein complexes that show high phosphorylation flux, sensitivity or drug/antibody targeting potential for further in silico or in vitro study. The extension leverages structural annotations from Section 4 and dynamic metrics from PhosKinTime to produce an integrated prioritisation.

## Why this is or is not close to the current framework

- **Classification:** downstream annotation layer / external pipeline integration.
- **Justification:** MD simulations operate on atomistic timescales and are outside the scope of the ODE-based model. However, the extension can use dynamic outputs (phosphorylation rates, sensitivities)[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation)[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) and structural annotations to guide MD simulation planning. It does not modify the dynamic model; therefore it is a downstream integration aimed at bridging PhosKinTime with molecular simulation pipelines.

## Current PhosKinTime assets this can reuse

- **Phosphorylation rate and kinase activity outputs:** used to identify sites with high flux[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation).
- **Sensitivity analysis:** identifies parameters whose perturbation most affects outcomes[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=); sites with high sensitivity may be prioritised.
- **Structural annotations:** generated by the structural extension; provide PDB/AlphaFold structures and per‑residue features.
- **Mapping tables:** `mapping.csv` and `psite_structural.csv` link dynamic entities to structure residues[\[12\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md#:~:text=,).
- **Configuration and export utilities:** to read dynamic outputs and write ranking tables.

## Required new scientific layer

- **Ranking criteria:** define a composite score for each phospho‑site or protein combining:

- Normalised phosphorylation rate magnitude[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation).

- Sensitivity score[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=).

- Structural accessibility (e.g., high SASA) and disorder from structural annotations.

- Potential druggability or antibody targeting (if mapped in other extensions). Weights for each criterion can be user‑defined.

- **MD readiness assessment:** check whether the structure covers the phospho‑site (no missing residues), whether the environment is conducive to MD (no unresolved loops) and whether post‑translational modifications (phosphorylation) can be parameterised with available force fields.

- **Preparation of simulation files:** generate coordinate (PDB) and topology (e.g., CHARMM or AMBER) inputs for the selected sites or proteins. Use existing tools (e.g., BioPandas or MDAnalysis) to extract segments and optionally apply phosphorylation modifications using libraries like ParmEd. Provide instructions or scripts for running MD in GROMACS or OpenMM, but do not run them within PhosKinTime.

- **Integration with docking or coarse‑grained simulation pipelines (optional):** prepare mutated or modified structures for further analysis.

## Realistic inputs

| Filename                        | Required columns                                                          | Optional columns                      | Units | Source                      | Notes                                                                          |
|---------------------------------|---------------------------------------------------------------------------|---------------------------------------|-------|-----------------------------|--------------------------------------------------------------------------------|
| `md_prioritization_params.toml` | `weights.rate`, `weights.sensitivity`, `weights.sasa`, `weights.disorder` | `weights.druggability`, `max_targets` |       | User-defined                | Defines how to weigh different criteria and how many targets to select.        |
| `psite_structural.csv`          | `gene`, `psite`, `rate`, `sensitivity`, `sasa`, `disorder`                | `druggability_score`                  |       | From structural extension   | Contains all metrics needed for ranking.                                       |
| `forcefield_params.csv`         | `residue`, `phosphorylated_topology`                                      |                                       |       | Force-field parameter files | To ensure that selected phospho‑sites are supported by chosen MD force fields. |
| `md_templates/`                 | Predefined MD system templates (e.g., water box, ion parameters)          |                                       |       | Provided by user            | Templates for MD simulation preparation.                                       |

## Realistic input sources

- **Outputs from PhosKinTime:** phosphorylation rates and sensitivity analyses[\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation)[\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=).
- **Structural extension outputs:** `psite_structural.csv` with SASA and disorder scores.
- **Force field libraries:** AMBER, CHARMM, GROMOS; parameters for phosphorylated residues.
- **MD tools:** MDAnalysis, BioPandas, ParmEd for structure manipulation; GROMACS or OpenMM for simulation templates.

## Outputs

| Output file            | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| `md_priority_list.csv` | Ranked list of sites or proteins with composite scores and selection rationale.                  |
| `md_ready_structures/` | Directory containing extracted PDB files of selected sites/proteins prepared for MD.             |
| `md_input_scripts/`    | Template simulation scripts (e.g., GROMACS `.mdp` files) with system setup parameters.           |
| `md_report.json`       | Metadata summarising ranking criteria, selected targets and provenance.                          |
| `md_plots/`            | Bar plots or scatter plots of composite scores vs. criteria weights to visualise prioritisation. |

## Proposed repository structure

    extensions/
      md_prioritization/
        __init__.py
        rank.py            # compute composite scores and produce ranking
        prepare.py         # generate MD-ready structure files
        export.py          # write rankings, scripts and plots
        config.py          # configuration schema
        cli.py
        tests/
          test_rank.py
          test_prepare.py

This extension depends on outputs from the structural extension; it should be placed under `extensions` and can be run separately.

## Configuration design

Add `[md_prioritization]` to `config.toml` or create a separate TOML file:

    [md_prioritization]
    # Weights for ranking criteria (must sum to 1)
    weights = {rate = 0.4, sensitivity = 0.3, sasa = 0.2, disorder = 0.1}
    # Maximum number of targets to select
    max_targets = 10
    # Force-field parameter file for phosphorylated residues
    forcefield_params = "data/forcefield_params.csv"
    # MD template directory
    templates = "data/md_templates/"
    # Output directory
    outdir = "results/md_prioritization/"

Users can adjust weights depending on the research question; the extension will normalise them.

## CLI design

- **Python module:** `phoskintime.extensions.md_prioritization.cli`.
- **Command:**


- python -m phoskintime md-prioritization --conf config.toml [--rank] [--prepare]


- **Pixi task:** add `md-prioritization` task with the same command.
- Flags: `--rank` generates the ranking list; `--prepare` creates MD-ready files for the top `max_targets`; `--plot` generates visualisations.
- Output directories follow `<outdir>/` with subdirectories `rankings`, `structures`, `scripts` and `plots`.

## Minimal viable implementation

- **Inputs:** `psite_structural.csv` with rate and sensitivity; user‑defined weights; basic force‑field parameter list for phosphorylated serine and tyrosine.
- **Algorithm:** Normalise each criterion; compute a weighted sum to obtain composite scores; sort sites; select top N; extract the corresponding residues and their structural environment (e.g., ±10 residues) using MDAnalysis; save as PDB files; generate placeholder MD simulation scripts referencing those PDBs.
- **Outputs:** `md_priority_list.csv` with ranking and scores; `md_ready_structures/` with extracted PDB segments; simple bar plot of scores.
- **Tests:** unit tests verifying scoring and ranking; integration test ensuring that extracted PDB segments contain the correct residues; test that missing force‑field parameters raise warnings.

## Full implementation roadmap

- **v0.1:** Implement ranking based on dynamic and structural metrics; prepare simple PDB extracts; generate template scripts; include tests.
- **v0.2:** Support ranking at protein level and include druggability/antibody targeting scores if available; add interactive plots; implement user-defined scoring functions.
- **v0.3:** Automate checking of force‑field compatibility; add optional energy minimisation using OpenMM; export results in formats ready for MD packages.
- **v0.4:** Integrate with docking or coarse‑grained simulation pipelines; provide example notebooks and cross‑validation on known benchmark proteins.
- **v1.0:** Mature extension with documentation and published use cases.

## Validation strategy

- **Unit tests:** verify correct computation of composite scores; ensure that ranking is reproducible and respects `max_targets` constraint.
- **Integration tests:** run the full pipeline on a small dataset; confirm that extracted PDB segments correspond to the intended residues and that template scripts reference the correct files.
- **Cross‑validation:** compare rankings to known importance of phospho‑sites from literature; confirm that high‑ranked sites often correspond to functional hotspots.
- **Regression tests:** ensure that modifications (e.g., weight changes) produce expected differences but not unintended side effects.

## Risks and failure modes

- **Subjective weighting:** composite scores depend on chosen weights; different users may prioritise different criteria.
- **Force-field limitations:** some modifications may not be supported; manual parameterisation may be needed.
- **Structural coverage:** structures may not include the region of interest, leading to ranking based on incomplete information.
- **Overinterpretation:** ranking does not guarantee functional importance; experimental validation is required.
- **Complex pipelines:** preparing MD-ready systems may require manual inspection; automation can fail on edge cases (e.g., hetero-oligomers).

## What not to do

- Do **not** run full-scale MD simulations within the extension; focus on preparing inputs and rankings.
- Do **not** claim that the top-ranked sites will always yield meaningful MD insights; they are suggestions based on available metrics.
- Do **not** assign equal weight to all criteria without justification; encourage users to tailor weights.
- Do **not** embed large MD templates or parameter files in the repository; provide links or instructions.
- Do **not** alter core PhosKinTime ODE models.

## Recommended priority

**Only as a downstream integration.** This extension relies on structural annotations and dynamic metrics; it does not feed back into the modelling. It should be developed after structural annotation and PK/PD modules are in place. Its value lies in guiding external MD studies rather than enhancing PhosKinTime core functionality.

# Recommended Implementation Order

1.  **Structural Bioinformatics Extension (Section 4)** — This is the most straightforward to implement because it operates as a separate post‑processing pipeline. It does not modify the core ODE model and relies on well‑established tools and publicly available databases. Structural annotation immediately enhances interpretability and enables downstream prioritisation of sites without requiring new experimental data.

2.  **Drug Dose‑Response / PK‑PD / QSP Extension (Section 1)** — Extending the model to include drug effects is highly relevant for pharmacological studies. The conceptual and implementation complexity is moderate; it leverages existing ODE infrastructure and parameter outputs. Implementing PK/PD will open the door to modelling other perturbations and is grounded in widely used PK models.

3.  **mRNA Vaccine / Antigen‑Expression / Immune‑Response Extension (Section 2)** — This extension requires adding new biological layers (exogenous mRNA, antigen translation and cytokine signalling) and depends on external kinetic data. While feasible, it introduces more assumptions and new state variables. It should follow once PK/PD modelling and structural annotation are stable.

4.  **Antibody Perturbation / Antibody‑Response Extension (Section 3)** — Modelling antibody binding kinetics is biologically interesting but demands detailed binding parameters and epitope mapping. Data scarcity and mapping uncertainties make this a lower priority. It can be implemented as a separate module after PK/PD and structural modules are mature.

5.  **Molecular Dynamics Simulation Prioritization Extension (Section 5)** — This is purely a downstream integration for ranking and preparing MD simulations. It depends on structural annotations and dynamic metrics, offering no direct feedback to the core model. It should be considered only after earlier extensions are in place and when there is specific interest in atomistic simulations.

[\[1\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=,factor%20scaling%20parameter) [\[2\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=predicted%20change%20should%20be%20interpreted,not%20as%20a%20standalone%20correlation) [\[3\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%60%60%60text%20TIME_POINTS_PROTEIN%20,phospho%20fitted%20trajectories) [\[4\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) [\[5\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) [\[9\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=) [\[10\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md#:~:text=%23%23%20Phosphorylation) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/main/networkmodel/INTERPRETATION.md>

[\[6\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/kinopt/README.md#:~:text=%2A%2Akinopt%2A%2A%20provides%20an%20end,for) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/main/kinopt/README.md>

[\[7\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,120%2C%20240%2C%20480%2C%20960) [\[8\]](https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml#:~:text=,0) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/global/config.toml>

[\[11\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md#:~:text=,mapping%20files%20for%20further%20analysis) [\[12\]](https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md#:~:text=,) raw.githubusercontent.com

<https://raw.githubusercontent.com/bibymaths/phoskintime/main/processing/README.md>
