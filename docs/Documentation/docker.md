# Docker Usage

This page describes how to build and run PhosKinTime inside a Docker container.

---

## Dockerfile Overview

The included `Dockerfile` uses `python:3.10-slim` as the base image and installs dependencies
via Poetry.

```dockerfile
FROM python:3.10-slim

RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root

COPY . /app

CMD ["poetry", "run", "phoskintime"]
```

**Default command:** `poetry run phoskintime` — runs the `phoskintime` CLI entry point.

---

## Build

```bash
docker build -t phoskintime .
```

---

## Run

### Run the default CLI

```bash
docker run --rm phoskintime --help
```

### Run with your data mounted

Mount your local `data/` directory and a results output directory into the container:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results_model" \
  phoskintime all
```

### Run a specific pipeline stage

```bash
# Preprocessing only
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results_model" \
  phoskintime prep

# Kinase optimization (local solver)
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results_model" \
  phoskintime kinopt --mode local
```

### Run the global model

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results_model_global" \
  -v "$PWD/config.toml:/app/config.toml" \
  --entrypoint poetry \
  phoskintime run phoskintime-global
```

### Pass a custom config file

```bash
docker run --rm \
  -v "$PWD/my_config.toml:/app/config.toml" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results_model" \
  phoskintime all
```

The container reads `config.toml` from `/app/` (the `WORKDIR`). Mount your own config file
to override the defaults.

---

## Launch the Dashboard

The Streamlit dashboard requires exposing a port:

```bash
docker run --rm \
  -v "$PWD/results_model_global:/app/results_model_global" \
  -p 8501:8501 \
  --entrypoint poetry \
  phoskintime run streamlit run run_dashboard.py -- --output-dir results_model_global
```

Then open `http://localhost:8501` in your browser.

---

## Limitations

- The Docker image uses `python:3.10-slim`, which satisfies the `>=3.10,<3.14` requirement.
- Multi-core parallelism inside Docker is limited by the host's CPU allocation. Use `--cpus`
  to set limits, and set `cores` in `config.toml` accordingly.
- The container does not persist results by default — always mount a host volume for output
  directories to avoid losing results when the container exits.
- Numba JIT compilation occurs on first run inside the container; subsequent runs reuse the
  compiled cache if the container filesystem is not discarded.
- The `Dockerfile` installs `--no-root` initially for caching, then copies the full source.
  If you modify source files locally, rebuild the image.
