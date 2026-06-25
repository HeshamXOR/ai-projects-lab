"""privacy-pii-vault core package.

A dependency-free (stdlib only) PII detection, reversible tokenization,
redaction-policy, and re-identification-risk toolkit. The public surface is
re-exported here for ergonomic imports::

    from core import Detector, Policy, PolicyEngine, Vault, k_anonymity
"""

from __future__ import annotations

from .detect import Detection, Detector, PIIType, detect, luhn_is_valid
from .gazetteer import (
    GIVEN_NAMES,
    ORG_TOKENS,
    SURNAMES,
    is_given_name,
    is_honorific,
    is_org_token,
    is_surname,
)
from .policy import (
    IssuedToken,
    Policy,
    PolicyAction,
    PolicyEngine,
    RedactionResult,
)
from .risk import EquivalenceClass, RiskReport, k_anonymity
from .tokenize import Vault, VaultEntry, derive_key, make_token

__all__ = [
    "Detection",
    "Detector",
    "PIIType",
    "detect",
    "luhn_is_valid",
    "GIVEN_NAMES",
    "SURNAMES",
    "ORG_TOKENS",
    "is_given_name",
    "is_surname",
    "is_org_token",
    "is_honorific",
    "Policy",
    "PolicyAction",
    "PolicyEngine",
    "RedactionResult",
    "IssuedToken",
    "Vault",
    "VaultEntry",
    "make_token",
    "derive_key",
    "k_anonymity",
    "RiskReport",
    "EquivalenceClass",
]

__version__ = "1.0.0"
