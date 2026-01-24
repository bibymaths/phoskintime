# Global Model Subpackage (`global_model`)

The `global_model` subpackage is the computational core of the PhosKinTime framework. It simulates the dynamic coupling between rapid kinase signaling and slower gene regulatory networks (GRNs) by solving a coupled, nonlinear ODE system, and provides tooling for calibration against time-resolved phosphoproteomics, proteomics, and transcriptomics data.

Key design goals:
- High-throughput simulation (Numba JIT RHS kernels; sparse topologies).
- Stable, bounded regulation dynamics (saturating transcription/translation modifiers).
- Optimization-ready API (parameter packing/unpacking, multi-objective loss aggregation).
- Explicit handling of missing network coverage (proxy logic for orphan TFs).

---

## Scope and Outputs

The model simulates three coupled layers per gene/protein entity:
- mRNA abundance for gene *g*: \(R_g(t)\)
- total protein abundance for gene *g*: \(P_g(t)\)
- phosphorylation-state abundances for protein *g*: \(P_{g,\cdot}(t)\) (model-dependent)

The core simulation returns time-indexed tables (typically pandas DataFrames) for:
- total protein \(P_g(t)\),
- mRNA \(R_g(t)\),
- phospho-states or phospho-observables (site-wise or state-wise, depending on topology).

---

## Notation and State Vector

We model \(G\) genes/proteins, \(K\) kinases, and (optionally) transcription factors (TFs) as a subset of proteins.

For each gene/protein \(g \in \{1,\dots,G\}\):
- \(R_g(t)\) is mRNA.
- \(P_g(t)\) is total protein.
- \(P_{g,s}(t)\) is a phosphorylation state or site-associated pool (definitions depend on the kinetic topology).

The full state vector is
$$
x(t) =
\begin{bmatrix}
R_1(t),\dots,R_G(t),\;
P_1(t),\dots,P_G(t),\;
\text{phospho states}(t)
\end{bmatrix}^\top
$$

We use:
- $u_k(t)$: observed kinase proxy input for kinase \(k\) (from MS abundance or derived activity proxy).
- $\beta_k$: learnable kinase gain scaling for kinase \(k\).
- $A$: sparse kinase-to-substrate-site adjacency (topology) from curated/constructed kinase–substrate links.
- $W$: TF-to-gene regulatory interaction matrix (GRN weights).

All equations below are written in continuous time; in practice, inputs are provided at discrete time points and integrated using a solver that respects these discrete changes (see [Implementation Notes](#implementation-notes)).

---

## Mathematical Framework

### Signaling Layer (Kinase Inputs)

Mass spectrometry typically measures abundance rather than catalytic activity. We therefore treat each kinase input as a scaled, learnable proxy:

For kinase $k$:
$$
\tilde{u}_k(t) = \beta_k \, u_k(t)
$$

For a given substrate protein $g$ and phosphorylation site (or transition) $s$, the total phosphorylation drive (“propensity”) is a sparse weighted sum over upstream kinases:
$$
\pi_{g,s}(t) = \sum_{k=1}^{K} A_{k\to(g,s)} \, \tilde{u}_k(t)
= \sum_{k=1}^{K} A_{k\to(g,s)} \, \beta_k \, u_k(t)
$$

Here $A_{k\to(g,s)}$ is typically binary (presence/absence) or may encode prior confidence/strength. In code, this is implemented as sparse lookups to avoid dense $K \times \sum_g S_g$ operations.

---

### Kinetic Topologies (Phosphorylation Dynamics)

The model supports multiple kinetic topologies to capture different biological mechanisms. Each topology specifies how phosphorylation state variables are defined and how mass flows between them.

Common parameters used across models:
- $\delta_{g,s}$: dephosphorylation (or reverse transition) rate for site/state $s$ on protein $g$.
- $\gamma_{g,s}$: degradation/turnover rate for site/state $s$ on protein $g$ (optional, when the state is explicitly tracked).
- $\rho_g$: basal protein turnover (for total protein).
- $\kappa_{g,s}$: effective forward phosphorylation rate scaling for site/state $s$ (often absorbed into $\pi_{g,s}$ depending on implementation).

The propensities $\pi_{g,s}(t)$ drive forward fluxes; dephosphorylation provides reverse fluxes.

#### A. Distributive / Independent (Model 0)

Assumption: sites behave independently; each site draws from a shared unphosphorylated pool $P_{g,0}(t)$ and produces a mono-phosphorylated pool $P_{g,s}(t)$.

State variables per protein $g$:
- $P_{g,0}(t)$: unphosphorylated pool.
- $P_{g,s}(t)$ for $s=1,\dots,S_g$: mono-phosphorylated pools per site.

Fluxes:
$$
F^{\text{phos}}_{g,s}(t) = \pi_{g,s}(t)\, P_{g,0}(t)
\quad,\quad
F^{\text{deph}}_{g,s}(t) = \delta_{g,s}\, P_{g,s}(t)
$$

ODEs:
$$
\frac{dP_{g,0}}{dt} =
\underbrace{\sum_{s=1}^{S_g} \delta_{g,s} P_{g,s}}_{\text{return from dephos}} -
\underbrace{\sum_{s=1}^{S_g} \pi_{g,s}(t) P_{g,0}}_{\text{forward phos}} -
\gamma_{g,0} P_{g,0}
$$

$$
\frac{dP_{g,s}}{dt} =
\pi_{g,s}(t) P_{g,0} - \delta_{g,s} P_{g,s}  - \gamma_{g,s} P_{g,s}
\quad,\quad s=1,\dots,S_g
$$

Observation mapping (common choice):
- site signal for $(g,s)$ is proportional to $P_{g,s}(t)$ (optionally normalized to $t=0$) in post-processing).

---

#### B. Sequential (Model 1)

Assumption: phosphorylation occurs in a strict order:
$$
P_{g,0} \rightarrow P_{g,1} \rightarrow \cdots \rightarrow P_{g,S_g}
$$
where $P_{g,i}$ denotes the state with exactly the first $i$ sites phosphorylated (or the $i$-th sequential stage).

Define transition propensities $\pi_{g,i}(t)$ for step $i: P_{g,i-1}\to P_{g,i}$.

Forward and reverse fluxes:
$$
F^{+}_{g,i}(t) = \pi_{g,i}(t)\, P_{g,i-1}(t)
\quad,\quad
F^{-}_{g,i}(t) = \delta_{g,i}\, P_{g,i}(t)
$$

ODEs:
$$
\frac{dP_{g,0}}{dt} =
\delta_{g,1} P_{g,1} - \pi_{g,1}(t) P_{g,0} - \gamma_{g,0} P_{g,0}
$$

For intermediate states $i=1,\dots,S_g-1$:
$$
\frac{dP_{g,i}}{dt} = \pi_{g,i}(t) P_{g,i-1} + \delta_{g,i+1} P_{g,i+1} - \delta_{g,i} P_{g,i} - \pi_{g,i+1}(t) P_{g,i} - \gamma_{g,i} P_{g,i}
$$

For the terminal state $i=S_g$:
$$
\frac{dP_{g,S_g}}{dt} =
\pi_{g,S_g}(t) P_{g,S_g-1} -
\delta_{g,S_g} P_{g,S_g} -
\gamma_{g,S_g} P_{g,S_g}
$$

---

#### C. Combinatorial / Hypercube (Model 2)

Assumption: all combinations of site phosphorylation are explicitly tracked, enabling synergistic/conditional effects. For $S_g$ sites, there are $2^{S_g}$ states.

Let $m \in \{0,1\}^{S_g}$ be a bitmask encoding a state (1 indicates phosphorylated). Denote the abundance of state $(m\) as \(P_{g,m}(t)$$.

For each site \(s\), define the neighboring states:
- \(m\) with bit \(s=0\) transitions to \(m^{(s,+)}\) where bit \(s\) is flipped to 1.
- \(m\) with bit \(s=1\) transitions back to \(m^{(s,-)}\) where bit \(s\) is flipped to 0.

A generic mass-action flux for a site flip:

$$
F^{(s,+)}_{g,m}(t) = \pi_{g,s}(t)\, P_{g,m}(t) \quad \text{if } m_s=0
$$ 

$$
F^{(s,-)}_{g,m}(t) = \delta_{g,s}\, P_{g,m}(t) \quad \text{if } m_s=1
$$

The ODE for each state \(m\) sums incoming and outgoing fluxes across all sites:

$$
\frac{dP_{g,m}}{dt} =
\sum_{s: m_s=1} \pi_{g,s}(t)\, P_{g,m^{(s,-)}} +
\sum_{s: m_s=0} \delta_{g,s}\, P_{g,m^{(s,+)}} -
\sum_{s: m_s=0} \pi_{g,s}(t)\, P_{g,m} -
\sum_{s: m_s=1} \delta_{g,s}\, P_{g,m} -
\gamma_{g,m}\, P_{g,m}
$$

Observation mapping can be site-wise:
$$
\text{site signal }(g,s) \propto \sum_{m: m_s=1} P_{g,m}(t)
$$

This topology is the most expressive but also the most expensive (\(O(2^{S_g})\) state scaling); it is only practical for small \(S_g\).

---

#### D. Saturating / Michaelis–Menten (Model 4)

Assumption: phosphorylation is enzyme-saturated at high substrate concentration, preventing “runaway” kinetics.

Let \(S(t)\) denote the relevant substrate pool for a given site/transition (often a state abundance such as \(P_{g,0}\) or \(P_{g,i}\)). The saturated forward flux for site \(s\) is:
$$
F^{\text{MM}}_{g,s}(t) =
V^{\max}_{g,s}(t)\, \frac{S(t)}{K_{M} + S(t)}
$$

We parameterize the effective maximum velocity as proportional to kinase drive:
$$
V^{\max}_{g,s}(t) = \pi_{g,s}(t)
$$

Using non-dimensionalized units, \(K_M\) is typically normalized to \(1.0\) unless explicitly estimated:
$$
F^{\text{MM}}_{g,s}(t) =
\pi_{g,s}(t)\, \frac{S(t)}{1 + S(t)}
$$

Example for an independent-site form (illustrative):
$$
\frac{dP_{g,s}}{dt} =
\pi_{g,s}(t)\, \frac{P_{g,0}}{1 + P_{g,0}} -
\delta_{g,s} P_{g,s} - \gamma_{g,s} P_{g,s}
$$

The saturating topology can be combined with sequential or distributive state definitions; the defining feature is the replacement of linear mass-action forward flux with the rational saturation term.

---

### Gene Expression Layer (GRN Coupling)

Protein levels feed back to gene expression via TF-mediated regulation. Let \(T\) be the set of TF indices, and let \(W \in \mathbb{R}^{G \times |T|}\) be the TF-to-gene interaction matrix, where \(W_{g,t}\) is positive for activation and negative for repression.

#### Transcriptional Regulation Input

Define the raw regulatory signal to gene \(g\):
$$
z_g(t) = \sum_{t \in T} W_{g,t}\, P_t(t)
$$

Optionally, a global scaling factor (`tf_scale`) is applied:
$$
\tilde{z}_g(t) = \alpha_{\text{tf}}\, z_g(t)
$$

#### Rational Hill / Bounded Modulation

We map the unbounded \(\tilde{z}_g(t)\) to a bounded synthesis modifier \(h_g(t)\) using a rational function for numerical stability and saturation.

First, input squashing:
$$
q_g(t) = \frac{\tilde{z}_g(t)}{1 + \left|\tilde{z}_g(t)\right|}
\quad\Rightarrow\quad
q_g(t)\in(-1,1)
$$

Then, activation vs repression mapping. A convenient rational form is:

Activation branch (\(q_g \ge 0\)):
$$
h_g(t) = 1 + \eta_g\, \frac{q_g(t)}{1 + q_g(t)}
$$

Repression branch (\(q_g < 0\)):
$$
h_g(t) = 1 - \xi_g\, \frac{-q_g(t)}{1 - q_g(t)}
$$

Here \(\eta_g \ge 0\) controls maximal fold-activation above baseline, and \(\xi_g \ge 0\) controls maximal fold-repression below baseline. This keeps \(h_g(t)\) bounded and avoids overflow for large \(|z_g|\).

#### mRNA and Protein Turnover

mRNA dynamics:
$$
\frac{dR_g}{dt} = s_g \, h_g(t) - d_g\, R_g
$$
where \(s_g\) is basal transcription and \(d_g\) is mRNA decay.

Protein dynamics (mass-action translation):
$$
\frac{dP_g}{dt} = k_g\, R_g - \rho_g\, P_g
$$
where \(k_g\) is translation rate and \(\rho_g\) is protein turnover.

For saturating translation (ribosome-limited), a bounded alternative is:
$$
\frac{dP_g}{dt} = k_g\, \frac{R_g}{1 + R_g} - \rho_g\, P_g
$$

---

## Biological Interpretations

### Proxy Logic for Orphan TFs

A common issue is missing phosphoproteomics coverage for TFs in the GRN. If a TF \(t\) is absent from observed/proxied inputs, leaving it constant can break feedback loops and reduce realism.

We therefore use a **proxy strategy**: if an orphan TF \(t\) is known (via the GRN or curated links) to regulate a kinase \(k\), we approximate TF activity as proportional to the kinase proxy:
$$
P_t(t) \propto \tilde{u}_k(t)
$$

In practice, this “rewiring” injects dynamics into otherwise disconnected TF nodes while preserving the sign/structure of feedback loops.

---

## Parameter Dictionary

The exact set of parameters depends on topology and configuration. Common parameters include:

Gene expression:
- $s_g$: basal transcription rate of gene $g$.
- $d_g$: mRNA decay rate $\mathrm{time}^{-1}$.
- $k_g$: translation rate per mRNA $\mathrm{time}^{-1}$.
- $\rho_g$: protein turnover $\mathrm{time}^{-1}$.
- $\alpha_{\text{tf}}$: global TF scaling (`tf_scale`).
- $\eta_g$, $\xi_g$: activation/repression strength (bounded modifiers).

Kinase signaling:
- $\beta_k$: kinase gain (global multiplier per kinase).
- $A_{k\to(g,s)}$: kinase-to-site topology/strength (sparse, usually fixed).

Phosphorylation dynamics:
- $\delta_{g,s}$: dephosphorylation rate for site/state.
- $\gamma_{g,\cdot}$: state-specific degradation/turnover (optional; may be shared).

---

## Module Architecture

The package is structured to separate data management, topology construction, physics kernels, numerical integration, and optimization.

| Module | Description |
| --- | --- |
| `network.py` | Topology engine. Defines the `Index` class mapping biological entities to state-vector indices. Implements proxy rewiring and builds sparse interaction structures. |
| `models.py` | Physics kernels. Numba JIT-compiled RHS functions for distributive, sequential, combinatorial, and saturating kinetics. |
| `solvers.py` | Numerical integration. Custom RK45-style adaptive solver with bucketed step control to handle piecewise-constant inputs without interpolation artifacts. |
| `simulate.py` | Simulation orchestration. Runs the ODE solve and produces measured/observable outputs aligned to experimental time points. |
| `optproblem.py` | Optimization wrapper. `GlobalODE_MOO` class compatible with `pymoo`; handles parameter unpacking, simulation, and loss aggregation. |
| `optimizer.py` | Strategy layer. Orchestrates global search (evolutionary / GA), iterative refinement (“zooming”), and hyperparameter tuning (e.g., via Optuna). |
| `lossfn.py` | Error metrics. JIT-compiled robust losses (Huber / Charbonnier), including optional weighting schemes for early time points. |
| `steadystate.py` | Initialization routines. Computes \(x_0\) by algebraic equilibrium or by mapping measured data at \(t=0\). |
| `sensitivity.py` | Analysis. Global sensitivity (e.g., Morris method) to quantify influential parameters (kinase gains, regulation strengths, etc.). |
| `config.py` | Central configuration loader for `config.toml` (paths, solver tolerances, bounds, time grids). |

---

## Configuration and Data Expectations

The typical workflow assumes:
1. `config.toml` defines dataset file paths, time point vectors per modality (protein / RNA / phospho), and solver tolerances.
2. Data loaders produce:
   - kinase proxy input table indexed by time (or aligned to nearest time buckets),
   - protein and RNA measurements for initialization and loss evaluation,
   - phospho measurements for site/state losses,
   - interaction maps (kinase–substrate and TF–target) to construct \(A\) and \(W\).

Practical requirements:
- Unique identifiers must be consistent across datasets (gene symbols / protein IDs).
- Time points should be strictly increasing; uneven spacing is supported.
- Missing values should be filtered or masked before optimization; loss functions typically include NaN guards.

---

## Implementation Notes

Numerical integration:
- Inputs \(u_k(t)\) are typically available at discrete experimental times. The solver therefore treats kinase inputs as piecewise-constant (or bucketed) in time, which avoids interpolation-induced artifacts when dynamics are fast relative to sampling.
- Adaptive RK45 controls local error using absolute/relative tolerances (configured in `config.toml`).

Performance:
- RHS kernels are Numba JIT-compiled; hot loops avoid Python object allocation.
- Sparse adjacency is stored in index lists / CSR-like structures to avoid dense matrix multiplies.
- Combinatorial topology is exponential in the number of sites per protein; use only for small \(S_g\).

Stability:
- Regulation modifiers are bounded by construction; this prevents explosive transcription/translation at large regulatory inputs.
- Optional normalization (e.g., fold-change to \(t=0\)) is applied in post-processing for comparability to MS logFC conventions.

Reproducibility:
- Keep `config.toml` and interaction maps under version control.
- Persist fitted parameter sets and seeds for optimization runs; log solver tolerances and objective weights.