"""Pytest configuration and shared test path setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``from pi_test_support import ...`` in test modules during collection.
_test_dir = str(Path(__file__).resolve().parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)
