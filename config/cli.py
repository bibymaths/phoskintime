"""
Command‑line entry point for the phoskintime pipeline.

Usage
--------------
# Come one level up from the package root, it should be the working directory
# (where you can see the project directory).

# run everything with the default (local) solver
python -m phoskintime all

# run only preprocessing
python -m phoskintime prep

# run tfopt with local flavour
python -m phoskintime tfopt --mode local

# run tfopt with evol flavour
python -m phoskintime tfopt --mode evol

# run kinopt with local flavour
python -m phoskintime kinopt --mode local

# run kinopt with evol flavour
python -m phoskintime kinopt --mode evol

# run the model
python -m phoskintime model
"""
from pathlib import Path
import subprocess as sp
import sys
import typer

# …/phoskintime
ROOT = Path(__file__).resolve().parent.parent
# current venv’s python
PY   = sys.executable

def _run(cmd: list[str]) -> None:
    """
    Echo + run from the package root.
    """
    typer.echo(f"$ {' '.join(cmd)}")
    sp.check_call([PY, "-m", *cmd], cwd=ROOT)

def _python_module(module: str, cfg: Path | None) -> list[str]:
    """
    Return `python -m module [--conf path]`.
    """
    cmd = [module]
    if cfg is not None:
        cmd += ["--conf", str(cfg)]
    return cmd

app = typer.Typer(help="CLI shortcuts for the phoskintime workflow")

@app.command()
def prep():
    """
    Preprocess data (processing.cleanup).
    """
    _run(["processing.cleanup"])

@app.command()
def tfopt(
    mode: str = typer.Option("local", help="local | evol"),
    conf: Path | None = typer.Option(
        None, "--conf", file_okay=True, dir_okay=False, writable=False,
        help="Path to TOML/YAML config. Uses defaults if omitted."
    ),
):
    """
    Transcription-Factor-mRNA Optimisation.
    """
    module = f"tfopt.{mode}"
    _run(_python_module(module, conf))

@app.command()
def kinopt(
    mode: str = typer.Option("local", help="local | evol"),
    conf: Path | None = typer.Option(
        None, "--conf", file_okay=True, dir_okay=False, writable=False,
        help="Path to TOML/YAML config. Uses defaults if omitted."
    ),
):
    """
    Kinase-Phosphorylation Optimization.
    """
    module = f"kinopt.{mode}"
    _run(_python_module(module, conf))

@app.command()
def model(
    conf: Path | None = typer.Option(
        None, "--conf", file_okay=True, dir_okay=False, writable=False,
        help="Path to model config file. Uses defaults if omitted."
    ),
):
    """
    Run the model (bin.main).
    """
    _run(_python_module("bin.main", conf))

@app.command()
def all(
    # propagate the same options to keep behaviour predictable
    tf_mode: str = typer.Option("local", help="tfopt mode: local | evol"),
    kin_mode: str = typer.Option("local", help="kinopt mode: local | evol"),
    tf_conf: Path | None = typer.Option(None, help="tfopt config file"),
    kin_conf: Path | None = typer.Option(None, help="kinopt config file"),
    model_conf: Path | None = typer.Option(None, help="model config file"),
):
    """
    Run every stage in sequence.
    """
    prep()
    tfopt.callback(mode=tf_mode, conf=tf_conf)
    kinopt.callback(mode=kin_mode, conf=kin_conf)
    model.callback(conf=model_conf)

if __name__ == "__main__":
    # Allow `python -m phoskintime ...`
    app()