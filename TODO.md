

# TODO: Fix TF regulation layer (currently missing / ineffective)

## Problem summary

The current transcription-factor (TF) regulation pathway is **effectively not present** in the model run, even though TF edges are loaded from a TF network file.

### What happens today

* `Index` defines modeled entities **only** from the kinase network column `interactions["protein"]`.
* `build_tf_matrix(tf_net, idx)` only adds TF edges if both:

  * `tf ∈ idx.proteins`
  * `target ∈ idx.proteins`
* In the dataset, TF names (e.g., `FOS`, `TP53`, `STAT3`) often **do not appear** in the kinase network’s modeled protein list, and many TF targets may also be absent.
* Therefore, most (often all) TF edges are filtered out → `tf_mat.nnz == 0`.
* As a result:

  * `TF_inputs = tf_mat.dot(P_vec) / tf_deg` becomes all zeros
  * `synth = Ai * (1 + tf_scale * TF_inputs[i])` collapses to `synth = Ai`
  * RNA dynamics becomes **purely constitutive** (no TF regulation)
  * The TF network file becomes a no-op and provides no biological constraint.

### Consequence

You are currently fitting a model where:

* kinases drive phosphorylation rates,
* RNA is produced with a constant basal rate `Ai`,
* and TF regulation is absent.
  This is a structural mismatch between the intended biology (“TFs regulate transcription”) and what is simulated.

---

## Required design decision

Pick one of the following TF modeling strategies. Each is valid, but they imply different code changes and biological interpretation.

### Option A — Protein→RNA feedback (minimal, consistent with current state space)

**Interpret TF effects as protein→transcription regulation where regulators are proteins already in the model.**

* Only allow TF edges where TF is a modeled protein (i.e., regulator ∈ `idx.proteins`).
* Filter TF network to model scope:

  * Keep only `(tf,target)` pairs where both are in `idx.proteins`.
* This makes the model internally consistent without introducing new states.

**Pros**

* Minimal changes
* Keeps current ODE structure and parameterization

**Cons**

* Biology is “protein abundance regulates transcription” (not explicit TF states)
* TFs not in kinase network are ignored

**Implementation TODO**

1. Add a “TF coverage report” printout:

   * number of TF edges loaded
   * number retained after filtering
   * list top missing TFs and missing targets
2. Apply explicit filtering in runner (before `build_tf_matrix`):

   * `tf_net = tf_net[tf in proteins and target in proteins]`
3. Add a warning/error if `tf_mat.nnz == 0`:

   * either abort run or print loud warning

---

### Option B — Expand model universe to include TF nodes as proteins (medium change)

**Add TFs as modeled nodes even if they are not present in the kinase network.**

* Extend `idx.proteins` to include union:

  * kinase-network proteins
  * TF sources
  * TF targets
* Then rebuild:

  * `p2i`, offsets, state dimension, etc.
* Update W construction rules:

  * New proteins may have 0 sites / no kinase inputs
* Define default parameters for newly added nodes.

**Pros**

* Preserves TF network structure better
* TF regulation can become active without new equation types

**Cons**

* State space grows
* Many new nodes have missing phosphorylation structure
* Adds parameter burden and risk of identifiability issues

**Implementation TODO**

1. Decide inclusion rule:

   * include all TFs? only TFs affecting observed targets? only TFs for observed genes?
2. Update `Index` to build proteins from union sets.
3. Ensure `build_W_parallel` handles proteins with:

   * no sites
   * no kinase edges
4. Define defaults/bounds for new proteins.
5. Re-run sanity checks (`tf_mat.nnz`, coverage stats).

---

### Option C — Explicit TF state variables (most biologically faithful, heavy change)

**Model TF as a separate species with its own dynamics, not equal to protein abundance.**

* Introduce TF states:

  * TF abundance or TF activity
* Link TF activity to protein and/or phospho state:

  * e.g., TF_active = f(phospho(TF)) or TF_active has its own ODE
* RNA synthesis depends on TF_active, not protein abundance.

**Pros**

* Best biological interpretation
* Can represent activation, repression, phospho-dependent TF activity

**Cons**

* Major refactor
* New parameters, new identifiability risks
* Requires curated assumptions (activation function, delays, etc.)

**Implementation TODO**

1. Define TF state representation:

   * abundance vs activity vs active/inactive split
2. Define how kinase/phospho drives TF activity.
3. Modify ODE RHS and indexing.
4. Update loss and measurement mapping accordingly.

---

## Must-have sanity checks (implement regardless of option)

These should be enforced to avoid silently running a broken TF layer again.

### 1) Coverage report at runtime

Print:

* `#TF edges input`
* `#TF edges retained`
* `tf_mat shape, nnz, density`
* top-N TFs not in model proteins
* top-N targets not in model proteins

### 2) Hard warning / failure if TF layer is empty

If the user provides a TF network and `tf_mat.nnz == 0`:

* either raise `RuntimeError`
* or require a flag like `--allow-empty-tf-net`

### 3) Document the actual interpretation in README

* If Option A: clearly state “protein abundance acts as transcriptional regulator”
* If Option B/C: explain what a TF node means (protein abundance vs TF activity)

---

## Secondary modeling gap: repression / sign of TF regulation

Currently TF influence is always additive and can explode:

```python
synth = Ai * (1 + tf_scale * u)
```

Needs:

* signed edges (+ activation, − repression)
* stable link function (to avoid negative synth or runaway growth), e.g.:

  * `synth = Ai * exp(tf_scale * u)`
  * or `synth = Ai * softplus(b0 + tf_scale * u)`
  * or clamp `u` and enforce positivity

**TODO**

1. Add TF edge sign support in TF matrix (from input column or defaults)
2. Switch synth function to always-positive, stable mapping
3. Add bounds / priors to prevent extreme TF_scale runaway

---

## Acceptance criteria (done means done)

This TODO is complete when:

1. Running with a TF network yields `tf_mat.nnz > 0` (unless explicitly allowed otherwise).
2. A runtime report shows meaningful overlap between TF net and modeled nodes.
3. RNA trajectories change when TF edges are perturbed (sanity test).
4. Documentation states which TF modeling strategy is used and what “TF_inputs” represents.

---
