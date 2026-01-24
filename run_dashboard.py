#! usr/bin/env python

"""
Run the PhosKinTime dashboard application.

This script sets up the necessary environment and launches the dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from global_model.dashboard_app import main

if __name__ == "__main__":
    main()
    print("Dashboard launched successfully!")
