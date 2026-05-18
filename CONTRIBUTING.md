# Contributing to phoskintime

Thank you for your interest in contributing to `phoskintime`. This document
describes how to set up the development environment, submit changes, and meet
the expectations for contributions to this scientific software project.

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/bibymaths/phoskintime.git
cd phoskintime
```

### Install with Pixi

This project uses [Pixi](https://prefix.dev/docs/pixi) for environment
management. Install Pixi if you have not already, then run:

```bash
# Default environment (runtime dependencies only)
pixi install

# Development environment (includes pytest)
pixi install -e dev

# Full environment (dev + docs + viz)
pixi install -e full
```

### Run the tests

```bash
pixi run -e dev test
```

To include a coverage report:

```bash
pixi run -e dev test-cov
```

### Verify project imports

```bash
pixi run check-import
```

### Build the documentation

```bash
pixi run -e docs docs-build
```

To serve the docs locally:

```bash
pixi run -e docs docs-serve
```

---

## Branch Naming

Use short, descriptive branch names prefixed by the type of change:

- `fix/short-description` — bug fixes
- `feat/short-description` — new features or modules
- `docs/short-description` — documentation-only changes
- `refactor/short-description` — internal refactoring without behavior change
- `ci/short-description` — workflow or configuration changes

---

## Commit Style

Write concise, imperative commit messages:

- `fix: correct NaN handling in sensitivity loop`
- `feat: add COBYQA optimizer wrapper`
- `docs: update contributing guide for Pixi setup`
- `test: add integration test for tfopt pipeline`

Avoid generic messages like `update`, `fix stuff`, or `WIP`.

---

## Pull Requests

1. Fork the repository and create your branch from `main`.
2. Ensure all tests pass: `pixi run -e dev test`.
3. If you changed or added documented code, rebuild docs and verify they
   compile: `pixi run -e docs docs-build`.
4. Open a pull request against `main` and fill in the PR template.
5. Link any relevant issue in the PR description.

### PR Checklist

- [ ] Tests pass (`pixi run -e dev test`)
- [ ] Docs build without errors if documentation was changed
- [ ] No generated result folders (e.g. `results_*/`, `logs/`) are committed
- [ ] No secrets, tokens, or local absolute paths are committed
- [ ] New dependencies are added intentionally to `pixi.toml` only
- [ ] Scientific assumptions or parameter choices are described in docstrings
      or comments

---

## Scientific Code Expectations

- **Reproducibility**: examples and tests must produce deterministic output
  (set random seeds where stochastic behavior is used).
- **Input/output clarity**: function signatures and docstrings should clearly
  state the expected shape, units, and type of inputs and outputs.
- **No hardcoded absolute paths**: use relative paths or configuration values.
- **Do not commit generated results**: output files from model runs belong in
  directories that are `.gitignore`d (e.g. `results_*/`).
- **Keep model parameters in configuration**: numerical parameters should be in
  `config.toml` or passed as arguments, not hardcoded in source.

---

## Dependency Policy

- Add all new runtime dependencies to `[dependencies]` or the appropriate
  `[feature.*.dependencies]` section of `pixi.toml`.
- Do not add `requirements.txt`, `environment.yml`, or conda/pip freeze dumps.
- Prefer conda-forge packages; use `[pypi-dependencies]` only when a package
  is not reliably available on conda-forge for all target platforms.

---

## Documentation

- Public functions and classes should have NumPy-style docstrings.
- Module-level `__doc__` strings are encouraged.
- Update relevant `.md` pages under `docs/` if behavior visible to users
  changes.

---

## Reporting Issues and Feature Requests

Use the GitHub issue tracker:
[https://github.com/bibymaths/phoskintime/issues](https://github.com/bibymaths/phoskintime/issues)

Issue templates are provided for bug reports, feature requests, and
documentation issues. Please use the appropriate template.
