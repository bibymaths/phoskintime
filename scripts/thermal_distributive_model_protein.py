"""Compatibility notice for retired standalone thermal protein script."""
from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "This standalone thermal script is retired from the active PhosKinTime "
        "JAXopt/Diffrax workflow. Use the centralized Diffrax solver utilities "
        "for new simulations."
    )


if __name__ == "__main__":
    main()
