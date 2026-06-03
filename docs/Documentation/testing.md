# Testing Notes

## Current validation commands

The documentation audit uses source inventory and syntax checks against the current networkmodel package.

```bash
python -m compileall -q networkmodel
griffe dump -s . -d google -L DEBUG networkmodel
```

## Current source checks

The public API inventory comes from `networkmodel/*.py`. Documentation should only mention public functions, classes, CLI flags, and configuration keys that appear in that inventory.

## What this module does not do

This page does not describe tests that are absent from the current repository source.
