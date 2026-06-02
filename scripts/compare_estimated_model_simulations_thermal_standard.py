"""Compatibility notice for retired thermal comparison script.

Current PhosKinTime networkmodel/protwise runs use JAXopt scalar optimization and
Diffrax Kvaerno ODE solving. Thermal comparison workflows should be regenerated
from saved scalar-objective outputs rather than this historical optimizer script.
"""
from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "This historical thermal comparison script is not part of the active "
        "PhosKinTime JAXopt/Diffrax workflow. Use networkmodel.runner outputs "
        "and mode-aware scalar result tables instead."
    )


if __name__ == "__main__":
    main()
