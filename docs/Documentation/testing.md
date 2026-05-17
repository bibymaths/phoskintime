# Testing

This page describes how to run the test suite and how tests are organized.

---

## Running Tests

```bash
pytest
```

From the project root. The test configuration is in `pytest.ini`.

With coverage reporting:

```bash
pytest --cov
```

---

## Current Test Coverage

The test suite is currently minimal. Only one test file exists:

### `tests/test_config.py`

Tests for `config/logconf.py`:

| Function | What it tests |
|---|---|
| `logger_handles_empty_log_directory` | Logger creates log dir if missing |
| `logger_does_not_duplicate_handlers` | Calling `setup_logger` twice does not duplicate handlers |
| `logger_handles_no_log_directory` | Logger works without a log dir (stream-only) |
| `logger_respects_custom_formatter` | Logger uses `ColoredFormatter` for all handlers |

> **Note:** These test functions are not prefixed with `test_`, so pytest will not collect them
> automatically. Prefix them with `test_` to make them active (e.g., `def test_logger_handles_...`).

---

## Current Limitations

- Most pipeline modules (`kinopt`, `tfopt`, `global_model`, `paramest`) have no automated tests.
- The test suite does not cover ODE solving, parameter estimation, optimization, or data I/O.
- Integration tests (end-to-end pipeline runs) do not exist.

---

## Recommended Pattern for Adding Tests

### Unit tests

Use pytest with `tmp_path` fixture for file-I/O tests:

```python
import pytest
from config.logconf import setup_logger

def test_logger_creates_log_dir(tmp_path):
    log_dir = str(tmp_path / "logs")
    setup_logger(name="test", log_dir=log_dir)
    assert (tmp_path / "logs").exists()
```

### What should be unit-tested

- Configuration loading (`config_loader.load()`) with a fixture `config.toml`
- Parameter bounds validation
- ODE right-hand side functions with known analytical inputs
- Loss function values for zero-error inputs
- Fréchet distance (`frechet.frechet_distance`) for trivial curves

### What should be integration-tested

- A minimal end-to-end run of `processing.cleanup` with a synthetic input CSV
- `kinopt.local` or `tfopt.local` with a small synthetic dataset
- `global_model.runner` with a tiny synthetic 2-gene network

### Running with Poetry

If using Poetry:

```bash
poetry run pytest
poetry run pytest --cov --cov-report=html
```

---

## pytest.ini

The project includes a `pytest.ini` at the root. Check it for any custom markers or
configuration that affects test discovery.
