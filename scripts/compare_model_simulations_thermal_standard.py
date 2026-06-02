"""Compatibility notice for retired thermal simulation comparison script."""
from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "This historical thermal simulation script is not part of the active "
        "PhosKinTime JAXopt/Diffrax workflow. Use Diffrax-backed networkmodel "
        "or protwise outputs for new comparisons."
    )


if __name__ == "__main__":
    main()
