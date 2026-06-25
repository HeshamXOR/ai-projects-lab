"""Pytest fixtures and sys.path setup.

Ensures the project root (containing the ``core`` package and ``app.py``) is
importable when tests run from any working directory.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
