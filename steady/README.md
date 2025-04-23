
# Steady-State Initializers for Phosphorylation Models

These scripts compute **biologically meaningful steady-state initial values** for different phosphorylation models, which are required as **starting points for ODE simulations**.

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

This is done using **constrained numerical optimization** (`scipy.optimize.minimize`) to solve a system of equations $f(\mathbf{y}) = 0$.

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
\begin{aligned}
A - B R &= 0 \\
C R - (D + \sum S_i) P + \sum P_i &= 0 \\
S_i P - (1 + D_i) P_i &= 0 \quad \forall i
\end{aligned}
$$

---

### 2. **Successive Model**

- Sites are phosphorylated in sequence
- Initial condition requires that **flow through the chain** is at equilibrium:
  - $P \rightarrow P_0 \rightarrow P_1 \rightarrow \dots \rightarrow P_n$

Steady-state means each conversion step:
- Has equal incoming and outgoing rates
- Balances intermediate accumulations

You solve:

$$
\text{Same logic, but with additional internal terms involving adjacent states}
$$

---

### 3. **Random Model**

- All possible phosphorylated combinations are treated as distinct states
- Total number of states = $2^n - 1$ (excluding unphosphorylated state)

You construct a system:

- One equation for $R$ and $P$
- One for each state $X_j$ (each subset of phosphorylated sites)
- For each state, compute net phosphorylation in/out, and degradation

Mathematically:

$$
\text{Each state } X_j: \quad \text{gain from P or other } X_k \quad - \quad \text{loss by phosphorylation, degradation, etc.} = 0
$$

This is a **sparse, coupled nonlinear system** where each subset has dynamic transitions with others.

---

## Output

Each function returns steady-state concentrations:

- $[R, P, P_1, ..., P_n]$ (for `distributive` and `successive`)
- $[R, P, X_1, ..., X_k]$ (for `random`, where $X_k$ are the subset states)

---