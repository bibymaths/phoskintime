# Steady-State Initializers for Phosphorylation Models

These scripts compute **biologically meaningful steady-state initial values** for different phosphorylation models,
which are required as **starting points for ODE simulations**.

Instead of guessing or using arbitrary initial values, we solve a **nonlinear system of equations** that ensures:

> **All time derivatives are zero at $t = 0$**  
> → i.e., the system is at equilibrium

---

## What Is Being Computed?

For each model, we're solving:

$$
\text{Find } y_0 \text{ such that } \frac{dy}{dt}\bigg|_{t=0} = 0
$$

where $\mathbf{y} = [R, P, \dots]$ are all species in the system.

This is done using **constrained numerical optimization** (`scipy.optimize.minimize`) to solve a system of
equations $f(\mathbf{y}) = 0$.

---

## Model-Specific Logic

### 1. **Distributive Model**

- Each site $i$ is phosphorylated independently
- Steady-state means:
    - mRNA synthesis balances degradation
    - Protein synthesis balances degradation and phosphorylation
    - Each phosphorylated state $P_i$ is in flux balance

You solve a nonlinear system:

$$
A - B R = 0  
$$ 
 
$$
C R - (D + \sum S_i) P + \sum P_i = 0  
$$ 
 
$$
S_i P - (1 + D_i) P_i = 0 \quad \forall i
$$

---

### 2. **Successive Model**

- Sites are phosphorylated in a fixed order:
  $P \to P_0 \to P_1 \to \dots \to P_n$
- Steady-state requires:
    - mRNA and protein production/degradation balance
    - Each intermediate state receives and passes flux without accumulation

You solve:

$$
A - B \cdot R = 0
$$

$$
C \cdot R - S_0 \cdot P + D_0 \cdot P_0 = 0
$$

$$
S_0 \cdot P - (S_1 + D_0) \cdot P_0 + D_1 \cdot P_1 = 0
$$

$$
S_1 \cdot P_0 - (S_2 + D_1) \cdot P_1 + D_2 \cdot P_2 = 0
$$

$$
\vdots
$$

$$
S_{n-1} \cdot P_{n-2} - (S_n + D_{n-1}) \cdot P_{n-1} + D_n \cdot P_n = 0
$$

$$
S_n \cdot P_{n-1} - D_n \cdot P_n = 0
$$

**Where:**

- $R$: mRNA concentration  
- $P$: unphosphorylated protein  
- $P_i$: protein with $i$ sites phosphorylated in sequence  
- $S_i$: phosphorylation rate from $P_{i-1} \to P_i$  
- $D_i$: degradation rate of $P_i$

---

### 3. **Random Model**

- All possible phosphorylated combinations are treated as distinct states
- Total number of states = $2^n - 1$ (excluding unphosphorylated state)

You construct a system:

- One equation for $R$ and $P$
- One for each state $X_j$ (each subset of phosphorylated sites)
- For each state, compute net phosphorylation in/out, and degradation

At steady state, each phosphorylated state $X_j$ satisfies:

$$
\frac{dX_j}{dt} = \sum_{k \in N_j^{\text{in}}} S_{k \to j} \cdot X_k \sum_{l \in N_j^{\text{out}}} S_{j \to l} \cdot X_j D_j \cdot X_j = 0
$$

**Where:**

- $X_j$: concentration of phosphorylation state $j$  
- $N_j^{\text{in}}$: set of states $k$ that transition into $X_j$  
- $N_j^{\text{out}}$: set of states $l$ that $X_j$ can transition into  
- $S_{a \rightarrow b}$: rate constant for transition from state $a$ to $b$ (e.g., phosphorylation/dephosphorylation)  
- $D_j$: degradation rate of state $X_j$ (depends on its phosphorylation pattern)

---

## Output

Each function returns steady-state concentrations:

- $[R, P, P_1, ..., P_n]$ (for `distributive` and `successive`)
- $[R, P, X_1, ..., X_k]$ (for `random`, where $X_k$ are the subset states)

---