#!/usr/bin/env python3
"""CLI entrypoint for running the cosmology analysis from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main


if __name__ == "__main__":
    main()
