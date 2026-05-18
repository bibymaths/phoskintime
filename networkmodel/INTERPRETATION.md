# PhoskinTime — Interpretation & Development Notes

This document captures how to interpret outputs from the global phosphorylation–transcription model and what to keep in
mind during further development.

---

## 1. Core Principle

This model is **dynamical and mechanistic**, not purely statistical.

- Outputs are **emergent behaviors** of an ODE system.
- Every observed effect is a **result of parameterized biochemical processes**:
    - kinase activity → phosphorylation
    - phosphorylation → protein state
    - protein state → transcriptional regulation

**Do not interpret outputs as correlations. Interpret them as propagated mechanistic effects.**

---

## 2. What a Knockout (KO) Means

A KO is implemented as a **parameter perturbation**, not node removal.

### Types:

- **Protein KO** → scaling `A_i` (synthesis)
- **Kinase KO** → scaling `Kmat` and `c_k`

### Implications:

- The node still exists in the system.
- Its **activity is suppressed**, not structurally deleted.
- Downstream effects arise through:
    - reduced phosphorylation flux
    - altered steady-state protein levels
    - modified TF activity

---

## 3. Interpretation of Outputs

### 3.1 Log2 Impact (KO / WT)

- Measures **steady-state shift**
- Interpretation:
    - `≈ 0` → no effect
    - `> 0` → upregulation after KO
    - `< 0` → downregulation after KO

⚠️ This is **not sensitivity**. It is endpoint behavior.

---

### 3.2 Sensitivity Score (Trajectory Difference)

- Computed as:

```

Σ |WT(t) - KO(t)|

```

- Captures **temporal perturbation magnitude**

Interpretation:

- High sensitivity → system strongly reacts over time
- Low sensitivity → system is robust or buffered

This is closer to a **functional importance metric** than steady-state change.

---

### 3.3 Δ Trajectories (WT − KO)

- Shows **signal loss propagation over time**
- Useful for:
- identifying early vs late responders
- distinguishing transient vs sustained effects

Interpretation:

- Early divergence → direct regulation
- Delayed divergence → indirect cascade

---

### 3.4 Kinase Drive (Σ W * Kt)

- Measures **total phosphorylation influence**
- Combines:
- network connectivity (`W`)
- dynamic activity (`Kt`)

Interpretation:

- High drive ≠ high expression
- High drive = strong **functional influence on phosphorylation layer**

---

### 3.5 Dominant Kinase per Site

- Identifies **which kinase controls each phospho-site**

Interpretation:

- High dominance count → control hub
- Low dominance → distributed regulation

---

## 4. Network Interpretation (Global + Influence Maps)

### 4.1 Edge Meaning

#### Signaling edges

```

kinase → protein
weight = Σ (beta * Kt)

```

- Represents **phosphorylation drive**
- Dynamic and context-dependent

#### Transcription edges

```

TF → target
weight = tf_scale * tf_mat * TF_level

```

- Represents **regulatory drive**
- Depends on protein abundance (not just topology)

---

### 4.2 What Edge Weight Is NOT

- Not probability
- Not confidence
- Not causal strength in isolation

It is:
> A **context-dependent instantaneous influence** under current system state

---

### 4.3 Δ Network (KO − WT)

This is the most important view.

Interpretation:

- Positive edge → strengthened under KO
- Negative edge → weakened under KO

Use this to identify:

- compensatory pathways
- broken signaling routes
- emergent rewiring

---

## 5. Seed Node Concept (Influence Maps)

- Seed = **starting point of cascade exploration**
- Usually:
    - the KO target (recommended default)

Interpretation:

- Depth 1 → direct targets
- Depth 2+ → propagated effects

⚠️ Changing seed changes **perspective**, not system behavior.

---

## 6. Time Matters

The system is **non-linear and time-dependent**.

### Key consequences:

- Early-time network ≠ steady-state network
- Edge weights evolve
- Dominant pathways can switch

### Best practice:

- Always compare:
    - early (minutes)
    - mid (transition)
    - late (steady-state)

---

## 7. Common Misinterpretations (Avoid These)

### ❌ “High edge weight = important pathway”

Not necessarily. It may be:

- transient
- redundant
- buffered downstream

---

### ❌ “No change = not relevant”

Wrong. Could mean:

- redundancy
- robustness
- compensation

---

### ❌ “KO effect is local”

Incorrect. Effects propagate through:

- phosphorylation cascades
- transcription loops

---

### ❌ “TF edges are static”

False. They depend on:

- protein levels (dynamic)
- system state

---

## 8. Development Guidelines

### 8.1 Always Separate

- **Model mechanics** (ODE, parameters)
- **Visualization layer** (graphs, plots)

Do not mix logic.

---

### 8.2 Use Hard System Reset

Always rebuild system for:

- WT
- KO
- sweep

```

sys_local, idx_local, ... = load_system()

```

Avoid state leakage.

---

### 8.3 Keep Time Grids Valid

ODE solvers require:

- strictly monotonic time arrays

Never reuse:

- unordered
- duplicated
- mixed grids

---

### 8.4 Scaling & Stability

- Edge weights can explode or vanish
- Always:
    - threshold (`min_abs_weight`)
    - cap (`max_edges`)

---

### 8.5 Performance Constraints

- gravis is heavy
- limit:
    - edges (~300–500)
    - timepoints (~10–15)

---

## 9. What This Model Is Good For

- Mechanistic hypothesis generation
- Pathway tracing
- KO impact analysis
- Multi-layer signaling + transcription coupling

---

## 10. What This Model Is NOT

- A statistical inference engine
- A causal discovery model
- A ground truth representation of biology

It is:
> A **structured dynamical hypothesis system**

---

## 11. Practical Workflow

1. Select KO
2. Inspect:
    - impact scatter
    - sensitivity
3. Check Δ trajectories
4. Analyze:
    - global network (Δ)
5. Drill down:
    - functional influence (seed = KO target)
6. Validate:
    - consistency across time

---

## 12. Final Guiding Rule

> Do not trust a single view.

Always triangulate:

- time dynamics
- steady state
- network structure

Only consistent signals across all three are meaningful.

---