# Script Entry Points

## Current entry points

The networkmodel runner exposes `main` in `runner.py`. The package initializer also exposes `main` and delegates to the runner.

## Command pattern

```bash
python -m networkmodel.runner --solver jaxopt --seed 42
```

## What this module does not do

This page does not document standalone scripts that are absent from the networkmodel public API inventory.
