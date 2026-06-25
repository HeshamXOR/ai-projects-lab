"""Policy engine: each action transforms detections correctly."""

from __future__ import annotations

from core.detect import Detector, PIIType
from core.policy import Policy, PolicyAction, PolicyEngine
from core.tokenize import Vault, derive_key

KEY = derive_key("policy-test-key")


def _engine():
    return PolicyEngine(vault=Vault(secret_key=KEY))


def test_redact_replaces_with_label():
    text = "Contact alice.johnson@example.com please."
    dets = Detector().detect(text)
    res = _engine().apply(text, dets, Policy(default_action=PolicyAction.REDACT))
    assert "[EMAIL]" in res.redacted_text
    assert "alice.johnson@example.com" not in res.redacted_text


def test_mask_keeps_last_four():
    text = "Card 4242 4242 4242 4242 used."
    dets = Detector().detect(text)
    res = _engine().apply(text, dets, Policy(default_action=PolicyAction.MASK, mask_keep=4))
    # Last four digits (4242) should remain visible; earlier digits masked.
    assert "*" in res.redacted_text
    assert res.redacted_text.rstrip(" used.").endswith("4242")


def test_hash_is_stable_and_irreversible():
    text = "SSN 123-45-6789 on file."
    dets = Detector().detect(text)
    eng = _engine()
    res1 = eng.apply(text, dets, Policy(default_action=PolicyAction.HASH))
    res2 = eng.apply(text, dets, Policy(default_action=PolicyAction.HASH))
    assert res1.redacted_text == res2.redacted_text  # stable
    assert "123-45-6789" not in res1.redacted_text


def test_per_type_overrides_apply():
    """Different types can get different actions in one pass."""
    text = "Email alice.johnson@example.com, card 4242 4242 4242 4242."
    dets = Detector().detect(text)
    policy = Policy(
        default_action=PolicyAction.REDACT,
        rules={PIIType.CREDIT_CARD: PolicyAction.MASK},
    )
    res = _engine().apply(text, dets, policy)
    assert "[EMAIL]" in res.redacted_text  # email redacted
    assert "4242" in res.redacted_text  # card masked, last4 kept


def test_offsets_preserved_with_length_change():
    """Right-to-left splicing keeps earlier spans correct even when the
    replacement length differs from the original."""
    text = "a alice.johnson@example.com b 192.168.1.42 c"
    dets = Detector().detect(text)
    res = _engine().apply(text, dets, Policy(default_action=PolicyAction.REDACT))
    # Both anchors and both labels survive in order.
    assert res.redacted_text.startswith("a [EMAIL] b [IP_ADDRESS] c")


def test_allow_leaves_value_untouched():
    text = "Email alice.johnson@example.com stays."
    dets = Detector().detect(text)
    res = _engine().apply(text, dets, Policy(default_action=PolicyAction.ALLOW))
    assert res.redacted_text == text
