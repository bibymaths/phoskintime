# Docker Notes

## Current scope

This page documents container use only at the level required to run the current Python source tree. The networkmodel code requires the same source files and configuration used by local execution.

## Command pattern

```bash
python -m networkmodel.runner --solver jaxopt --seed 42
```

The command invokes the implemented runner entry point. Container images must provide the Python dependencies needed by JAX, Diffrax, JAXopt, pandas, NumPy, matplotlib, and Streamlit when dashboard rendering is used.

## What this module does not do

This page does not define a separate optimizer backend for containers. It does not document deployment behavior that is absent from the source files.
