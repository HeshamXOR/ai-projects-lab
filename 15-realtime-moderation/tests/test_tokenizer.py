"""Tests for the from-scratch tokenizer."""

from core.tokenizer import Tokenizer


def test_lowercase_and_split():
    tok = Tokenizer()
    assert tok.split("Hello World") == ["hello", "world"]


def test_punctuation_stripped():
    tok = Tokenizer()
    assert tok.split("Hello, world! How's it?") == ["hello", "world", "how's", "it"]


def test_accent_stripping():
    tok = Tokenizer()
    assert tok.split("naïve café") == ["naive", "cafe"]


def test_leet_folding():
    tok = Tokenizer(fold_leet=True)
    # "st0pid" -> "stopid" (0->o), "1diot" -> "idiot" (1->i)
    assert tok.split("st0pid 1diot") == ["stopid", "idiot"]


def test_leet_can_be_disabled():
    tok = Tokenizer(fold_leet=False)
    assert tok.split("h3llo") == ["h3llo"]


def test_bigrams():
    tok = Tokenizer(ngram_range=(1, 2))
    out = tok.tokenize("the quick fox")
    assert "the" in out and "quick" in out and "fox" in out
    assert "the quick" in out and "quick fox" in out


def test_ngram_only_bigrams():
    tok = Tokenizer(ngram_range=(2, 2))
    assert tok.tokenize("a b c") == ["a b", "b c"]


def test_empty_string():
    tok = Tokenizer()
    assert tok.split("") == []
    assert tok.tokenize("") == []


def test_callable_interface():
    tok = Tokenizer()
    assert tok("Hi there") == ["hi", "there"]
