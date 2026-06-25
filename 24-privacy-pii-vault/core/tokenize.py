"""Reversible, format-preserving pseudonymization with a keyed vault.

WHY this module exists
----------------------
Redaction destroys data; tokenization *replaces* it with a realistic but fake
surrogate while letting an authorized party restore the original later. Two
properties matter:

  * **Format preservation** -- a 16-digit card must map to a 16-digit token,
    an email to an email-shaped string, a phone to a phone-shaped string. This
    keeps downstream systems (which validate formats) working on tokenized
    data, and stops the token itself from leaking that "something was here".

  * **Determinism with a secret key** -- the same input under the same key
    always yields the same token, so referential integrity is preserved
    across records (two rows with the same SSN tokenize identically) without
    storing a lookup just to be consistent. We derive token digits/characters
    from ``HMAC-SHA256(key, type || value)``. HMAC is a keyed PRF: without the
    key the output is computationally unlinkable to the input, and changing
    the key changes every token.

Reversibility is provided by a :class:`Vault` that stores the encrypted
mapping token -> original. (The token is a deterministic *pseudonym*; the
vault is the authoritative reverse index, mirroring how real tokenization
services separate the pseudonym from the recoverable secret.)

SECURITY CAVEAT
---------------
The vault here keeps plaintext originals in memory keyed by token. In
production you would persist them in an HSM-backed store and encrypt at rest.
HMAC determinism means token equality reveals input equality *to a key
holder*; this is intentional (referential integrity) but is a known
re-identification vector across joined datasets -- combine with the k-anonymity
scorer before releasing tokenized data.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from .detect import PIIType

_DIGITS = "0123456789"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _hmac_stream(key: bytes, label: str, value: str) -> bytes:
    """Return the raw HMAC-SHA256 digest for ``label || value`` under ``key``.

    The label (PII type) is mixed in so that the *same* string tokenizes
    differently across types -- e.g. an SSN-shaped value used as an SSN versus
    as a phone yields distinct surrogates, preventing cross-type collisions.
    """
    msg = f"{label}\x00{value}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).digest()


def _digit_sequence(key: bytes, label: str, value: str, count: int) -> str:
    """Deterministically derive ``count`` decimal digits from the HMAC.

    We re-hash with an incrementing counter (HMAC-DRBG style) so we can emit
    arbitrarily long digit strings while staying deterministic and uniform.
    """
    out: list[str] = []
    counter = 0
    while len(out) < count:
        block = _hmac_stream(key, f"{label}:{counter}", value)
        for byte in block:
            # Rejection-free mod-10 is slightly biased but acceptable for a
            # pseudonym; uniformity is not a security requirement here.
            out.append(_DIGITS[byte % 10])
            if len(out) == count:
                break
        counter += 1
    return "".join(out)


def _alpha_sequence(key: bytes, label: str, value: str, count: int) -> str:
    """Deterministically derive ``count`` lowercase letters from the HMAC."""
    out: list[str] = []
    counter = 0
    while len(out) < count:
        block = _hmac_stream(key, f"{label}:alpha:{counter}", value)
        for byte in block:
            out.append(_LOWER[byte % 26])
            if len(out) == count:
                break
        counter += 1
    return "".join(out)


def _fpe_card(key: bytes, value: str) -> str:
    """Format-preserving surrogate for a credit card.

    Preserves the exact length and separator layout (spaces/dashes) of the
    input, replaces each digit with a derived digit, then *fixes the final
    digit* so the surrogate also passes Luhn -- a downstream card validator
    will accept the token.
    """
    digits = [c for c in value if c.isdigit()]
    fake_digits = list(_digit_sequence(key, "CARD", value, len(digits)))

    # Recompute the last digit so the whole thing satisfies Luhn.
    def luhn_total(ds: list[int]) -> int:
        total = 0
        for idx, d in enumerate(reversed(ds)):
            if idx % 2 == 1:
                doubled = d * 2
                total += doubled - 9 if doubled > 9 else doubled
            else:
                total += d
        return total

    body = [int(d) for d in fake_digits[:-1]] + [0]
    check = (10 - luhn_total(body) % 10) % 10
    fake_digits[-1] = str(check)

    # Re-insert separators in their original positions.
    result: list[str] = []
    di = 0
    for ch in value:
        if ch.isdigit():
            result.append(fake_digits[di])
            di += 1
        else:
            result.append(ch)
    return "".join(result)


def _fpe_generic_digits(key: bytes, label: str, value: str) -> str:
    """Replace each digit in ``value`` with a derived digit, keeping layout.

    Used for phones, SSNs and IPs: punctuation, parentheses and separators are
    preserved so the surrogate keeps the same recognizable shape.
    """
    digits = [c for c in value if c.isdigit()]
    fake = _digit_sequence(key, label, value, len(digits))
    result: list[str] = []
    di = 0
    for ch in value:
        if ch.isdigit():
            result.append(fake[di])
            di += 1
        else:
            result.append(ch)
    return "".join(result)


def _fpe_ip(key: bytes, value: str) -> str:
    """Format-preserving IPv4 surrogate with each octet in 0-255."""
    octets = value.split(".")
    digest = _hmac_stream(key, "IP", value)
    new_octets = []
    for idx, _ in enumerate(octets):
        # Two bytes per octet -> value mod 256 keeps it a valid octet.
        b = digest[(idx * 2) % len(digest)]
        new_octets.append(str(b % 256))
    return ".".join(new_octets[: len(octets)])


def _fpe_email(key: bytes, value: str) -> str:
    """Format-preserving email surrogate.

    Keeps an ``@`` and the original TLD so the result is still a syntactically
    valid email of similar shape; the local part and the domain label are
    replaced by derived alphanumeric strings of the same length.
    """
    if "@" not in value:
        return _alpha_sequence(key, "EMAIL", value, max(1, len(value)))
    local, _, domain = value.partition("@")
    # Preserve TLD (last dot-segment); randomize the rest of the domain.
    if "." in domain:
        dom_label, _, tld = domain.rpartition(".")
    else:
        dom_label, tld = domain, "com"
    new_local = _alpha_sequence(key, "EMAIL_LOCAL", value, max(1, len(local)))
    new_dom = _alpha_sequence(key, "EMAIL_DOM", value, max(1, len(dom_label)))
    return f"{new_local}@{new_dom}.{tld}"


def _fpe_person(key: bytes, value: str) -> str:
    """Surrogate personal name: replace each alphabetic word, keep capitals."""
    words = value.split()
    out = []
    for w in words:
        alpha = _alpha_sequence(key, "PERSON", value + w, max(1, len(w)))
        out.append(alpha.capitalize())
    return " ".join(out)


def _fpe_org(key: bytes, value: str) -> str:
    """Surrogate organization name preserving any trailing org keyword."""
    words = value.split()
    out = []
    for w in words:
        stripped = w.lower().strip(".,")
        # Keep recognizable org suffixes so it still reads like a company.
        from . import gazetteer

        if stripped in gazetteer.ORG_TOKENS:
            out.append(w)
        else:
            alpha = _alpha_sequence(key, "ORG", value + w, max(1, len(w)))
            out.append(alpha.capitalize())
    return " ".join(out)


def _fpe_date(key: bytes, value: str) -> str:
    """Surrogate date: shift digits while preserving the layout/separators."""
    return _fpe_generic_digits(key, "DATE", value)


# Dispatch table mapping a PII type to its format-preserving generator.
_FPE_DISPATCH = {
    PIIType.CREDIT_CARD: _fpe_card,
    PIIType.PHONE: lambda k, v: _fpe_generic_digits(k, "PHONE", v),
    PIIType.SSN: lambda k, v: _fpe_generic_digits(k, "SSN", v),
    PIIType.IP_ADDRESS: _fpe_ip,
    PIIType.EMAIL: _fpe_email,
    PIIType.PERSON: _fpe_person,
    PIIType.ORG: _fpe_org,
    PIIType.DATE: _fpe_date,
}


def make_token(key: bytes, pii_type: PIIType, value: str) -> str:
    """Produce a deterministic, format-preserving surrogate for ``value``.

    Same ``(key, pii_type, value)`` always yields the same token; a different
    key yields a different token. The output preserves the structural format
    of the input for the given type.
    """
    generator = _FPE_DISPATCH.get(pii_type)
    if generator is None:
        # Fallback: alpha string of same length.
        return _alpha_sequence(key, pii_type.value, value, max(1, len(value)))
    return generator(key, value)


@dataclass
class VaultEntry:
    """A reversible mapping record stored in the vault."""

    token: str
    original: str
    pii_type: PIIType


@dataclass
class Vault:
    """Thread-safe in-memory reverse index for detokenization.

    The vault is keyed by ``(pii_type, token)`` so that identical surrogate
    strings under different types never collide. ``tokenize`` is idempotent:
    re-tokenizing the same value returns the cached token and does not create
    duplicate entries.

    SECURITY: stores plaintext originals -- see module docstring. Treat the
    vault as the crown-jewel secret store.
    """

    secret_key: bytes
    _by_token: Dict[tuple, VaultEntry] = field(default_factory=dict)
    _by_value: Dict[tuple, str] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def tokenize(self, pii_type: PIIType, value: str) -> str:
        """Return the surrogate for ``value`` and record the reverse mapping."""
        with self._lock:
            vkey = (pii_type, value)
            if vkey in self._by_value:
                return self._by_value[vkey]
            token = make_token(self.secret_key, pii_type, value)
            entry = VaultEntry(token=token, original=value, pii_type=pii_type)
            self._by_token[(pii_type, token)] = entry
            self._by_value[vkey] = token
            return token

    def detokenize(self, pii_type: PIIType, token: str) -> Optional[str]:
        """Return the original value for ``token`` or ``None`` if unknown."""
        with self._lock:
            entry = self._by_token.get((pii_type, token))
            return entry.original if entry is not None else None

    def detokenize_any(self, token: str) -> Optional[str]:
        """Look up ``token`` across all types (first match wins)."""
        with self._lock:
            for (_, tok), entry in self._by_token.items():
                if tok == token:
                    return entry.original
            return None

    def size(self) -> int:
        """Number of distinct mappings currently stored."""
        with self._lock:
            return len(self._by_token)


def derive_key(passphrase: str, salt: bytes = b"pii-vault-v1") -> bytes:
    """Derive a 32-byte key from a passphrase using PBKDF2-HMAC-SHA256.

    Stretching the passphrase makes brute-forcing the key materially harder
    than hashing it once; the fixed salt keys the deployment namespace.
    """
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000)
