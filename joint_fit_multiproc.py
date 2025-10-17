#!/usr/bin/env python3
"""Compatibility wrapper for the refactored cosmology CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Delegate execution to :mod:`src.main`."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
