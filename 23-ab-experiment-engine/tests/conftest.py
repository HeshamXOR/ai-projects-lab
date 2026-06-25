"""Pytest configuration: ensure the project root is importable.

Adds the repository root to ``sys.path`` so ``import core`` and ``import app``
work regardless of where pytest is invoked from.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
