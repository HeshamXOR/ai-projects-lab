"""Tests that the from-scratch Naive Bayes classifier actually learns."""

import math

import pytest

from core.naive_bayes import MultinomialNaiveBayes
from core.tokenizer import Tokenizer


def _train_on_bundled(dataset, tokenizer):
    texts, labels = dataset
    docs = [tokenizer.tokenize(t) for t in texts]
    model = MultinomialNaiveBayes(alpha=1.0).fit(docs, labels)
    return model, docs, labels


def test_trains_to_high_accuracy(dataset, tokenizer):
    model, docs, labels = _train_on_bundled(dataset, tokenizer)
    acc = model.score(docs, labels)
    assert acc >= 0.95, f"train accuracy too low: {acc}"


def test_learns_classes_and_vocab(dataset, tokenizer):
    model, _, _ = _train_on_bundled(dataset, tokenizer)
    assert set(model.classes_) == {"toxic", "clean"}
    assert len(model.vocabulary_) > 20


def test_generalizes_to_held_out_clean(dataset, tokenizer):
    model, _, _ = _train_on_bundled(dataset, tokenizer)
    # Held-out clean examples not in the training set.
    held_out = [
        "thank you for the lovely gift",
        "the meeting is scheduled for monday morning",
        "i enjoyed the concert very much",
    ]
    for text in held_out:
        assert model.predict(tokenizer.tokenize(text)) == "clean", text


def test_generalizes_to_held_out_toxic(dataset, tokenizer):
    model, _, _ = _train_on_bundled(dataset, tokenizer)
    held_out = [
        "you are a stupid worthless idiot",
        "i hate you you pathetic loser",
        "shut up you disgusting moron",
    ]
    for text in held_out:
        assert model.predict(tokenizer.tokenize(text)) == "toxic", text


def test_probabilities_normalize(dataset, tokenizer):
    model, _, _ = _train_on_bundled(dataset, tokenizer)
    proba = model.predict_proba(tokenizer.tokenize("you are an idiot"))
    assert abs(sum(proba.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in proba.values())


def test_toxic_probability_higher_for_toxic(dataset, tokenizer):
    model, _, _ = _train_on_bundled(dataset, tokenizer)
    p_toxic = model.predict_proba(tokenizer.tokenize("you worthless idiot"))["toxic"]
    p_clean = model.predict_proba(tokenizer.tokenize("thank you kindly"))["toxic"]
    assert p_toxic > 0.8
    assert p_clean < 0.2


def test_laplace_smoothing_handles_unseen():
    # Two tiny classes; an unseen token must not crash and must keep proba valid.
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit([["good", "nice"], ["bad", "awful"]], ["pos", "neg"])
    proba = model.predict_proba(["totallyunseenword"])
    assert abs(sum(proba.values()) - 1.0) < 1e-9


def test_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha=0.0).fit([["a"]], ["x"])


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit([["a"], ["b"]], ["x"])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MultinomialNaiveBayes().predict(["a"])


def test_log_proba_matches_manual_computation():
    # A hand-checkable case: verify log-prior is log(class fraction).
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit([["a"], ["a"], ["b"]], ["x", "x", "y"])
    # 2 of 3 docs are class x.
    assert math.isclose(model.log_prior_["x"], math.log(2 / 3))
    assert math.isclose(model.log_prior_["y"], math.log(1 / 3))
