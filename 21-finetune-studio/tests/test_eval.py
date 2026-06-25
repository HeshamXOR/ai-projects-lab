"""Tests that PROVE the eval metrics match hand-computed values.

  (5) perplexity == exp(mean NLL) on a tiny known logit/label case
  (6) token accuracy matches a hand value
  plus: IGNORE_INDEX masking, exact-match, streaming Evaluator consistency.
"""

import math

import torch

from core.data import IGNORE_INDEX
from core.eval import (
    Evaluator,
    gather_token_log_probs,
    mean_negative_log_likelihood,
    perplexity,
    sequence_exact_match,
    token_accuracy,
)


# (5) perplexity on a hand-computable case -----------------------------------
def test_perplexity_hand_computed():
    # Vocab size 2. Two positions. Logits chosen so log-probs are clean.
    # Position 0: logits [0, 0] -> softmax [0.5, 0.5]; gold=0 -> log p = ln 0.5.
    # Position 1: logits [ln 3, 0] -> softmax [0.75, 0.25]; gold=1 -> log p = ln 0.25.
    logits = torch.tensor([[[0.0, 0.0], [math.log(3.0), 0.0]]])  # [1, 2, 2]
    labels = torch.tensor([[0, 1]])                               # [1, 2]

    lp = gather_token_log_probs(logits, labels)
    expected_lp = torch.tensor([math.log(0.5), math.log(0.25)])
    assert torch.allclose(lp, expected_lp, atol=1e-6)

    nll = mean_negative_log_likelihood(logits, labels)
    expected_nll = -0.5 * (math.log(0.5) + math.log(0.25))
    assert abs(nll - expected_nll) < 1e-6

    ppl = perplexity(logits, labels)
    assert abs(ppl - math.exp(expected_nll)) < 1e-6
    # Sanity: should equal sqrt(1/(0.5*0.25)) = sqrt(8).
    assert abs(ppl - math.sqrt(8.0)) < 1e-6


# (6) token accuracy hand value ----------------------------------------------
def test_token_accuracy_hand_value():
    # 4 positions; argmax predictions vs labels: 3 of 4 correct -> 0.75.
    logits = torch.tensor(
        [[
            [2.0, 0.0],  # argmax 0
            [0.0, 2.0],  # argmax 1
            [2.0, 0.0],  # argmax 0
            [0.0, 2.0],  # argmax 1
        ]]
    )
    labels = torch.tensor([[0, 1, 1, 1]])  # position 2 wrong (pred 0, gold 1)
    assert abs(token_accuracy(logits, labels) - 0.75) < 1e-9


# masking: IGNORE_INDEX positions excluded from both metrics ------------------
def test_ignore_index_masking():
    logits = torch.tensor([[[0.0, 0.0], [math.log(3.0), 0.0]]])
    labels = torch.tensor([[IGNORE_INDEX, 1]])  # only second position counts
    lp = gather_token_log_probs(logits, labels)
    assert lp.numel() == 1
    assert abs(lp.item() - math.log(0.25)) < 1e-6
    # accuracy: only position 1, pred argmax = 0 != gold 1 -> 0.0
    assert token_accuracy(logits, labels) == 0.0


def test_exact_match():
    preds = ["The Answer", "paris", "no"]
    refs = ["the answer", "Paris", "yes"]
    # First two match after normalization; last doesn't -> 2/3.
    assert abs(sequence_exact_match(preds, refs) - 2.0 / 3.0) < 1e-9


def test_streaming_evaluator_matches_single_pass():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 7)
    labels = torch.randint(0, 7, (3, 5))
    labels[0, 0] = IGNORE_INDEX

    ev = Evaluator()
    # Feed in two chunks; corpus-level result must equal one-shot computation.
    ev.update(logits[:2], labels[:2])
    ev.update(logits[2:], labels[2:])
    res = ev.result()

    ref_ppl = perplexity(logits, labels)
    ref_acc = token_accuracy(logits, labels)
    assert abs(res.perplexity - ref_ppl) < 1e-5
    assert abs(res.token_accuracy - ref_acc) < 1e-9
