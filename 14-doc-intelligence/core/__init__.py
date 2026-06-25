"""Document intelligence core.

A from-scratch invoice/receipt extraction engine. The heavy lifting -- field
extraction, line-item parsing, normalization and confidence scoring -- is
implemented here using only the Python standard library.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import confidence, fields, lineitems, normalize, pipeline  # noqa: F401

__all__ = ["confidence", "fields", "lineitems", "normalize", "pipeline", "__version__"]
