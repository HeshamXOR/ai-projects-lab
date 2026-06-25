"""Detection precision/recall on a labeled sample paragraph.

Proves the detector finds known emails, phones, SSNs, cards, IPs and names,
and that precision & recall on the labeled sample clear a quality bar.
"""

from __future__ import annotations

from core.detect import Detector, PIIType, luhn_is_valid

# A labeled paragraph. SAMPLE is the raw text; GOLD lists the (type, value)
# pairs a correct detector should find. 4242-4242-4242-4242 is a Luhn-valid
# test card.
SAMPLE = (
    "Dr. Alice Johnson emailed alice.johnson@example.com from the office. "
    "Her direct line is (415) 555-0132 and her SSN is 123-45-6789. "
    "She paid with card 4242 4242 4242 4242 on 2023-05-17. "
    "The server at 192.168.1.42 logged the transaction. "
    "Bob Smith works at Acme Technologies Inc."
)

GOLD = [
    (PIIType.PERSON, "Alice Johnson"),
    (PIIType.EMAIL, "alice.johnson@example.com"),
    (PIIType.PHONE, "(415) 555-0132"),
    (PIIType.SSN, "123-45-6789"),
    (PIIType.CREDIT_CARD, "4242 4242 4242 4242"),
    (PIIType.DATE, "2023-05-17"),
    (PIIType.IP_ADDRESS, "192.168.1.42"),
    (PIIType.PERSON, "Bob Smith"),
    (PIIType.ORG, "Acme Technologies Inc."),
]


def _found_set(detections):
    return {(d.type, d.value) for d in detections}


def test_detector_finds_each_gold_type():
    """Every gold PII type should be represented in the detections."""
    dets = Detector().detect(SAMPLE)
    found_types = {d.type for d in dets}
    for pii_type, _ in GOLD:
        assert pii_type in found_types, f"missing type {pii_type}"


def test_detection_recall_meets_bar():
    """Recall over the labeled gold spans must be >= 0.8."""
    dets = Detector().detect(SAMPLE)
    found = _found_set(dets)

    def matches(gold_value: str) -> bool:
        # A gold item counts as found if some detection's value covers it
        # (handles trailing punctuation / span differences).
        for d in dets:
            if gold_value in d.value or d.value in gold_value:
                return True
        return False

    hits = sum(1 for _, value in GOLD if matches(value))
    recall = hits / len(GOLD)
    assert recall >= 0.8, f"recall {recall:.2f} below 0.8 (found {found})"


def test_detection_precision_reasonable():
    """Precision should be high: few spurious detections on clean text."""
    dets = Detector().detect(SAMPLE)
    gold_values = [v for _, v in GOLD]

    def is_true_positive(d) -> bool:
        return any(d.value in g or g in d.value for g in gold_values)

    if not dets:
        raise AssertionError("no detections at all")
    tp = sum(1 for d in dets if is_true_positive(d))
    precision = tp / len(dets)
    assert precision >= 0.7, f"precision {precision:.2f} below 0.7"


def test_no_false_card_on_invalid_luhn():
    """A 16-digit run that fails Luhn must not be reported as a card."""
    text = "Order number 1234 5678 9012 3456 was shipped."
    dets = Detector().detect(text)
    cards = [d for d in dets if d.type == PIIType.CREDIT_CARD]
    # 1234567890123456 is not Luhn-valid, so no card detection expected.
    assert not luhn_is_valid("1234567890123456")
    assert cards == [], f"unexpected card detection: {cards}"


def test_honorific_creates_person_for_novel_name():
    """An honorific should yield a PERSON even for an out-of-gazetteer name."""
    dets = Detector().detect("Please contact Dr. Zyxqua about the results.")
    persons = [d for d in dets if d.type == PIIType.PERSON]
    assert any("Zyxqua" in d.value for d in persons)
