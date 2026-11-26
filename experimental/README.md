# experimental/

Code kept for reference but quarantined from the default workflows (KNOWN_ISSUES.md N12).

- `pinn/` — PINN / NeuralODE extensions of the network model (hybrid and pure neural right-hand sides).
  The networkmodel runner refuses `enable_pinn`; nothing in the default workflow imports this package.
  Rules for any future use: no PINNs in mechanistic experiments, and solver failures must never be masked.
  It is not covered by the docs navigation.
