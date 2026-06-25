"""Tokenization: round-trips, format preservation, determinism, and Luhn."""

from __future__ import annotations

from core.detect import PIIType, luhn_is_valid
from core.policy import Policy, PolicyAction, PolicyEngine
from core.tokenize import Vault, derive_key, make_token

KEY_A = derive_key("alpha-secret")
KEY_B = derive_key("beta-secret")

EMAIL = "alice.johnson@example.com"
CARD = "4242 4242 4242 4242"
PHONE = "(415) 555-0132"
SSN = "123-45-6789"
IP = "192.168.1.42"


def test_luhn_accepts_valid_rejects_invalid():
    """Luhn validator accepts a known-valid card, rejects an invalid one."""
    assert luhn_is_valid("4242 4242 4242 4242")  # valid test card
    assert luhn_is_valid("4111111111111111")  # valid Visa test number
    assert not luhn_is_valid("4242 4242 4242 4243")  # last digit broken
    assert not luhn_is_valid("1234567890123456")  # arbitrary, invalid
    assert not luhn_is_valid("123")  # too short


def test_email_token_is_email_shaped():
    """An email token preserves the email shape and the original TLD."""
    token = make_token(KEY_A, PIIType.EMAIL, EMAIL)
    assert "@" in token
    local, _, domain = token.partition("@")
    assert local and "." in domain
    assert token.endswith(".com")
    assert token != EMAIL


def test_card_token_is_16_digits_and_luhn_valid():
    """A 16-digit card tokenizes to 16 digits that also pass Luhn."""
    token = make_token(KEY_A, PIIType.CREDIT_CARD, CARD)
    digits = [c for c in token if c.isdigit()]
    assert len(digits) == 16
    assert luhn_is_valid(token)
    assert token != CARD


def test_phone_and_ssn_preserve_layout():
    """Phone/SSN tokens keep the same separator layout and digit count."""
    pt = make_token(KEY_A, PIIType.PHONE, PHONE)
    assert sum(c.isdigit() for c in pt) == sum(c.isdigit() for c in PHONE)
    assert pt[0] == "(" and ")" in pt  # layout preserved
    st = make_token(KEY_A, PIIType.SSN, SSN)
    assert st.count("-") == 2
    assert [len(p) for p in st.split("-")] == [3, 2, 4]


def test_ip_token_octets_in_range():
    """An IP token is a valid dotted-quad with octets 0-255."""
    token = make_token(KEY_A, PIIType.IP_ADDRESS, IP)
    parts = token.split(".")
    assert len(parts) == 4
    assert all(0 <= int(p) <= 255 for p in parts)


def test_determinism_same_key_same_token():
    """Same input + same key -> identical token, every time."""
    for typ, val in [
        (PIIType.EMAIL, EMAIL),
        (PIIType.CREDIT_CARD, CARD),
        (PIIType.SSN, SSN),
    ]:
        a = make_token(KEY_A, typ, val)
        b = make_token(KEY_A, typ, val)
        assert a == b


def test_determinism_different_key_different_token():
    """A different key yields a different token for the same input."""
    for typ, val in [
        (PIIType.EMAIL, EMAIL),
        (PIIType.CREDIT_CARD, CARD),
        (PIIType.SSN, SSN),
    ]:
        assert make_token(KEY_A, typ, val) != make_token(KEY_B, typ, val)


def test_vault_round_trip_via_policy_engine():
    """TOKENIZE then detokenize restores the original exactly."""
    vault = Vault(secret_key=KEY_A)
    engine = PolicyEngine(vault=vault)
    text = f"Email {EMAIL}, card {CARD}, ssn {SSN}."
    from core.detect import Detector

    dets = Detector().detect(text)
    policy = Policy(default_action=PolicyAction.TOKENIZE)
    result = engine.apply(text, dets, policy)

    assert result.issued_tokens, "expected tokens to be issued"
    for issued in result.issued_tokens:
        restored = vault.detokenize(issued.type, issued.token)
        # The original substring lives at the recorded span in the source.
        start, end = issued.original_span
        assert restored == text[start:end]


def test_vault_idempotent_tokenization():
    """Re-tokenizing the same value returns the same token, no duplicate."""
    vault = Vault(secret_key=KEY_A)
    t1 = vault.tokenize(PIIType.EMAIL, EMAIL)
    t2 = vault.tokenize(PIIType.EMAIL, EMAIL)
    assert t1 == t2
    assert vault.size() == 1


def test_detokenize_unknown_returns_none():
    """Detokenizing a token that was never issued returns None."""
    vault = Vault(secret_key=KEY_A)
    assert vault.detokenize(PIIType.EMAIL, "nobody@nowhere.com") is None
