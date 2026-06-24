"""Proofs for the from-scratch NLP core."""

import numpy as np

from core.tfidf import TfidfVectorizer
from core.logreg import LogisticRegression, sigmoid
from core.skillgraph import SkillGraph


def test_tfidf_downweights_common_terms():
    docs = ["the cat the cat", "the dog the dog", "kubernetes scaling"]
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    # rows are L2-normalized
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9)
    # "the" appears in 2/3 docs -> lower idf than "kubernetes" (1/3 doc)
    assert vec.idf[vec.vocab["the"]] < vec.idf[vec.vocab["kubernetes"]]


def test_sigmoid_stable_on_extremes():
    z = np.array([-1000.0, 0.0, 1000.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] < 1e-6 and abs(p[1] - 0.5) < 1e-9 and p[2] > 1 - 1e-6


def test_logreg_learns_separable_data():
    rng = np.random.default_rng(0)
    # two clearly separated gaussian blobs
    X0 = rng.normal(-2, 0.5, (100, 2))
    X1 = rng.normal(+2, 0.5, (100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    clf = LogisticRegression(lr=0.5, epochs=300).fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.98
    # loss decreased
    assert clf.history[-1] < clf.history[0]


def test_skillgraph_suggests_adjacent():
    docs = [
        {"python", "docker", "kubernetes"},
        {"python", "docker", "aws"},
        {"docker", "kubernetes", "aws"},
        {"python", "pandas"},
    ]
    g = SkillGraph().build(docs)
    # someone with docker+python should be pointed toward kubernetes/aws
    suggestions = dict(g.suggest({"docker", "python"}, top=3))
    assert "kubernetes" in suggestions or "aws" in suggestions
