"""Policy engine: apply a per-type transform to every detected entity.

WHY this module exists
----------------------
Detection answers *where* the PII is; policy answers *what to do about it*.
Different data has different handling requirements -- an audit log may need
emails fully removed, but a fraud workflow needs the last four digits of a
card visible while the rest is masked, and an analytics pipeline needs
reversible tokens so records can later be re-joined. Encoding these choices as
data (a :class:`Policy` mapping PII type -> :class:`PolicyAction`) keeps the
transform logic in one auditable place and makes the behavior testable.

The engine applies transforms **right-to-left over the detections** so that
replacing a span never invalidates the character offsets of spans earlier in
the string -- a subtle correctness requirement when surrogate length differs
from the original length.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .detect import Detection, PIIType
from .tokenize import Vault


class PolicyAction(str, Enum):
    """What to do with a detected entity.

    REDACT:
        Replace with a type label, e.g. ``[EMAIL]``. Irreversible, no leakage.
    MASK:
        Keep the last few characters, replace the rest with ``*``. Useful for
        human verification ("ending in 4242") while hiding the bulk.
    TOKENIZE:
        Replace with a deterministic, format-preserving, *reversible* surrogate
        stored in the vault. Restorable via /detokenize.
    HASH:
        Replace with a truncated SHA-256 hex digest. Irreversible but stable,
        so equal inputs map to equal hashes (pseudonymous join key).
    ALLOW:
        Leave the value untouched (explicit opt-out for a type).
    """

    REDACT = "REDACT"
    MASK = "MASK"
    TOKENIZE = "TOKENIZE"
    HASH = "HASH"
    ALLOW = "ALLOW"


@dataclass
class Policy:
    """A redaction policy: a default action plus per-type overrides.

    Parameters
    ----------
    default_action:
        Action applied to any PII type without an explicit rule.
    rules:
        Mapping of :class:`PIIType` to :class:`PolicyAction`.
    mask_keep:
        Number of trailing characters MASK leaves visible (default 4).
    hash_length:
        Hex characters retained from the SHA-256 digest for HASH.
    """

    default_action: PolicyAction = PolicyAction.REDACT
    rules: Dict[PIIType, PolicyAction] = field(default_factory=dict)
    mask_keep: int = 4
    hash_length: int = 12

    def action_for(self, pii_type: PIIType) -> PolicyAction:
        """Return the action configured for ``pii_type`` (or the default)."""
        return self.rules.get(pii_type, self.default_action)

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, str],
        default_action: str = "REDACT",
        mask_keep: int = 4,
        hash_length: int = 12,
    ) -> "Policy":
        """Build a Policy from plain strings (as received over the API)."""
        rules = {
            PIIType(k.upper()): PolicyAction(v.upper()) for k, v in mapping.items()
        }
        return cls(
            default_action=PolicyAction(default_action.upper()),
            rules=rules,
            mask_keep=mask_keep,
            hash_length=hash_length,
        )


@dataclass
class IssuedToken:
    """Record of a TOKENIZE transform, returned to the caller for audit."""

    type: PIIType
    original_span: tuple
    token: str


@dataclass
class RedactionResult:
    """Outcome of applying a policy to a piece of text."""

    redacted_text: str
    detections: List[Detection]
    issued_tokens: List[IssuedToken] = field(default_factory=list)


def _mask(value: str, keep: int) -> str:
    """Mask all but the last ``keep`` *alphanumeric* characters of ``value``.

    Non-alphanumeric characters (``@``, ``.``, ``-``, spaces) are preserved so
    the masked form keeps a recognizable shape, e.g.
    ``****@****.com`` is avoided in favor of preserving structure.
    """
    if keep <= 0:
        return "*" * len(value)
    alnum_positions = [i for i, c in enumerate(value) if c.isalnum()]
    if len(alnum_positions) <= keep:
        # Too short to mask meaningfully; mask everything but last char.
        keep = max(0, len(alnum_positions) - 1)
    keep_from = alnum_positions[len(alnum_positions) - keep] if keep else len(value)
    out = []
    for i, c in enumerate(value):
        if c.isalnum() and i < keep_from:
            out.append("*")
        else:
            out.append(c)
    return "".join(out)


def _hash(value: str, length: int) -> str:
    """Return a stable truncated SHA-256 hex digest of ``value``."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:length]


@dataclass
class PolicyEngine:
    """Applies a :class:`Policy` to text using a :class:`Vault` for tokens."""

    vault: Vault

    def apply(self, text: str, detections: List[Detection], policy: Policy) -> RedactionResult:
        """Transform ``text`` per ``policy``, returning the redacted result.

        Detections are processed in descending start order so that each
        in-place replacement preserves the offsets of not-yet-processed,
        earlier spans.
        """
        issued: List[IssuedToken] = []
        # Work on a mutable copy; splice replacements right-to-left.
        result = text
        ordered = sorted(detections, key=lambda d: d.start, reverse=True)

        for det in ordered:
            action = policy.action_for(det.type)
            replacement = self._transform(det, action, policy, issued)
            result = result[: det.start] + replacement + result[det.end :]

        # Issued tokens were appended right-to-left; restore document order.
        issued.sort(key=lambda t: t.original_span[0])
        return RedactionResult(
            redacted_text=result,
            detections=sorted(detections, key=lambda d: d.start),
            issued_tokens=issued,
        )

    def _transform(
        self,
        det: Detection,
        action: PolicyAction,
        policy: Policy,
        issued: List[IssuedToken],
    ) -> str:
        """Compute the replacement string for one detection under ``action``."""
        if action is PolicyAction.ALLOW:
            return det.value
        if action is PolicyAction.REDACT:
            return f"[{det.type.value}]"
        if action is PolicyAction.MASK:
            return _mask(det.value, policy.mask_keep)
        if action is PolicyAction.HASH:
            return f"[{det.type.value}:{_hash(det.value, policy.hash_length)}]"
        if action is PolicyAction.TOKENIZE:
            token = self.vault.tokenize(det.type, det.value)
            issued.append(
                IssuedToken(
                    type=det.type,
                    original_span=(det.start, det.end),
                    token=token,
                )
            )
            return token
        raise ValueError(f"unhandled policy action: {action}")
