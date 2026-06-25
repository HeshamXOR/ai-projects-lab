"""Proofs for the from-scratch core: intent classifier, TF-IDF, dialog FSM,
endpointing/barge-in, and slot extraction."""

import numpy as np

from core.intent import IntentClassifier, TfidfVectorizer
from core.slots import fill_slots
from core.dialog import DialogManager, State
from core.endpointing import (
    Endpointer,
    EndpointConfig,
    detect_barge_in,
    frames_from_energies,
)


# ---------------------------------------------------------------------------
# (1) Intent classifier trains and classifies held-out phrasings.
# ---------------------------------------------------------------------------

_TRAIN = [
    ("hello there", "greet"),
    ("hi good morning", "greet"),
    ("hey assistant", "greet"),
    ("goodbye", "goodbye"),
    ("bye see you", "goodbye"),
    ("that is all bye", "goodbye"),
    ("book a flight to paris", "book_flight"),
    ("i want to fly to london", "book_flight"),
    ("reserve a flight to tokyo", "book_flight"),
    ("what is the weather in berlin", "check_weather"),
    ("weather forecast for madrid", "check_weather"),
    ("is it raining in rome", "check_weather"),
]


def _trained_classifier() -> IntentClassifier:
    texts = [t for t, _ in _TRAIN]
    labels = [l for _, l in _TRAIN]
    return IntentClassifier(n_iters=1200, learning_rate=0.5).fit(texts, labels)


def test_intent_classifier_classifies_heldout_phrasings():
    clf = _trained_classifier()
    # phrasings not seen verbatim in training
    assert clf.predict("hello") == "greet"
    assert clf.predict("see you later bye") == "goodbye"
    assert clf.predict("book a flight to berlin") == "book_flight"
    assert clf.predict("what is the weather in paris") == "check_weather"


def test_predict_proba_is_distribution():
    clf = _trained_classifier()
    proba = clf.predict_proba("book a flight to rome")
    assert abs(sum(proba.values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in proba.values())
    assert max(proba, key=proba.get) == "book_flight"


# ---------------------------------------------------------------------------
# (2) TF-IDF properties: rarer terms get higher idf; vectors are L2-normalized.
# ---------------------------------------------------------------------------


def test_tfidf_idf_higher_for_rare_terms():
    docs = [
        "the cat sat",
        "the dog sat",
        "the bird sat",
        "the rareword appears once",
    ]
    vec = TfidfVectorizer().fit(docs)
    common = vec.vocabulary_["the"]      # in every doc
    rare = vec.vocabulary_["rareword"]   # in one doc
    assert vec.idf_[rare] > vec.idf_[common]


def test_tfidf_rows_are_l2_normalized():
    docs = ["the cat sat on the mat", "a dog barked loudly"]
    matrix = TfidfVectorizer().fit_transform(docs)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# (3) Dialog FSM: missing slot -> AWAITING_SLOT + follow-up; then proceeds.
# ---------------------------------------------------------------------------


def test_dialog_missing_slot_asks_followup_then_proceeds():
    dm = DialogManager()
    # book_flight requires destination + date; this turn gives only destination
    r1 = dm.handle("book_flight", "book a flight to paris")
    assert r1.state == State.AWAITING_SLOT
    assert "date" in r1.missing
    assert r1.response  # asked a follow-up

    # follow-up supplies the date; frame completes -> needs confirmation
    r2 = dm.handle("book_flight", "june 25")
    assert r2.state == State.CONFIRMING
    assert r2.slots.get("destination") == "paris"
    assert r2.slots.get("date")

    # user confirms -> RESPONDING with the booking
    r3 = dm.handle("book_flight", "yes")
    assert r3.state == State.RESPONDING
    assert "paris" in r3.response.lower()


def test_dialog_complete_intent_responds_directly():
    dm = DialogManager()
    # check_weather needs only city, supplied up front
    r = dm.handle("check_weather", "what is the weather in london")
    assert r.state == State.RESPONDING
    assert "london" in r.response.lower()
    assert r.missing == []


def test_dialog_goodbye_ends():
    dm = DialogManager()
    r = dm.handle("goodbye", "bye now")
    assert r.state == State.DONE


# ---------------------------------------------------------------------------
# (4) Endpointing: end-of-utterance after silence; barge-in detection.
# ---------------------------------------------------------------------------


def test_endpointing_detects_end_after_silence():
    # speech for ~0.5s (loud frames), then a long silence
    energies = [0.1] * 5 + [0.0] * 12  # 0.1s frames
    frames = frames_from_energies(energies, frame_period=0.1)
    ep = Endpointer(EndpointConfig(energy_threshold=0.02, silence_timeout=0.8, min_speech_frames=3))
    end_ts = ep.process(frames)
    assert end_ts is not None
    # end should fall after at least 0.8s of trailing silence past last voiced frame
    assert end_ts >= 0.4 + 0.8 - 1e-9


def test_endpointing_no_end_without_enough_silence():
    energies = [0.1] * 5 + [0.0] * 3  # only 0.3s of trailing silence
    frames = frames_from_energies(energies, frame_period=0.1)
    ep = Endpointer(EndpointConfig(silence_timeout=0.8, min_speech_frames=3))
    assert ep.process(frames) is None


def test_barge_in_detected():
    # the user starts speaking: a sustained run of voiced frames
    energies = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1]
    frames = frames_from_energies(energies, frame_period=0.1)
    result = detect_barge_in(frames, EndpointConfig(energy_threshold=0.02, barge_in_min_frames=3))
    assert result.detected
    assert result.timestamp is not None


def test_no_barge_in_on_brief_noise():
    energies = [0.0, 0.1, 0.0, 0.1, 0.0]  # flapping, never 3 in a row
    frames = frames_from_energies(energies, frame_period=0.1)
    result = detect_barge_in(frames, EndpointConfig(barge_in_min_frames=3))
    assert not result.detected


# ---------------------------------------------------------------------------
# (5) Slot extraction pulls a date / number / city.
# ---------------------------------------------------------------------------


def test_slot_extraction_pulls_city_and_date():
    result = fill_slots("check_weather", "weather in london tomorrow")
    assert result.slots.get("city") == "london"
    assert result.slots.get("date") == "tomorrow"
    assert result.missing == []


def test_slot_extraction_pulls_number():
    result = fill_slots("set_alarm", "set an alarm at 7am")
    assert result.slots.get("time") == "7am"


def test_slot_extraction_reports_missing_required():
    result = fill_slots("book_flight", "i want to book a flight")
    # neither destination nor date present
    assert "destination" in result.missing
    assert "date" in result.missing
