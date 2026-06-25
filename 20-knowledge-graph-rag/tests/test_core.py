"""Proofs for the from-scratch extractor, knowledge graph, and graph-augmented
retrieval. The final test demonstrates graph expansion genuinely beating the
lexical baseline on a multi-hop question."""

from core.extract import extract, detect_entities, extract_relations
from core.graph import KnowledgeGraph
from core.retrieval import Retriever


# ---------------------------------------------------------------- extractor
def test_extractor_pulls_entities_and_relation():
    text = "Acme acquired Beta in 2022."
    result = extract(text)
    assert "Acme" in result.entities
    assert "Beta" in result.entities
    assert ("Acme", "acquired", "Beta") in result.triples


def test_extractor_multiword_and_role_relation():
    text = "Jane Smith is the CEO of Acme Corp."
    ents = detect_entities(text)
    assert "Jane Smith" in ents
    assert "Acme Corp" in ents
    triples = extract_relations(text, ents)
    # the role pattern should produce an is_ceo_of triple
    assert any(rel == "is_ceo_of" for _h, rel, _t in triples)


def test_extractor_cooccurrence_counts():
    text = "Acme acquired Beta. Acme also partnered with Beta."
    result = extract(text)
    # Acme and Beta co-occur in both sentences
    key = tuple(sorted(("Acme", "Beta")))
    assert result.cooccurrence.get(key, 0) >= 2


# ------------------------------------------------------------------- graph
def _build_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_relation("Acme", "acquired", "Beta", weight=2.0)
    g.add_relation("Beta", "owns", "Gamma", weight=2.0)
    g.add_relation("Gamma", "based_in", "Berlin", weight=2.0)
    g.add_relation("Delta", "founded", "Epsilon", weight=2.0)  # disconnected
    return g


def test_graph_neighbors():
    g = _build_graph()
    nbrs = {n for (n, _r, _w) in g.neighbors("Acme")}
    assert "Beta" in nbrs
    # undirected view sees the incoming edge to Beta from Acme
    und = {n for (n, _r, _w) in g.neighbors("Beta", undirected=True)}
    assert "Acme" in und and "Gamma" in und


def test_graph_shortest_path_multihop():
    g = _build_graph()
    path = g.shortest_path("Acme", "Berlin")
    assert path == ["Acme", "Beta", "Gamma", "Berlin"]


def test_graph_shortest_path_unreachable():
    g = _build_graph()
    assert g.shortest_path("Acme", "Epsilon") is None


def test_subgraph_for_query_connected():
    g = _build_graph()
    sub = g.subgraph_for_query("Tell me about Acme", k=2)
    assert "Acme" in sub and "Beta" in sub and "Gamma" in sub
    # 2 hops should not yet reach Berlin (3 hops away)
    assert "Berlin" not in sub
    # disconnected node never appears
    assert "Epsilon" not in sub


# ------------------------------------------- graph-augmented BEATS baseline
def test_graph_augmentation_beats_baseline():
    """Crafted corpus: the answer passage shares NO terms with the question but
    is reachable through the graph (Acme -> acquired -> Beta -> Gamma).

    Question mentions only 'Acme'. The passage describing Gamma's product shares
    no query terms, so baseline lexical retrieval misses it. Graph expansion
    pulls in 'Beta' and 'Gamma' as terms and surfaces it.
    """
    passages = [
        "Acme is a large holding company headquartered downtown.",
        "Gamma manufactures industrial robotics for warehouses.",  # the answer
        "The cafeteria introduced a seasonal menu in spring.",
    ]
    g = KnowledgeGraph()
    g.add_relation("Acme", "acquired", "Beta", weight=2.0)
    g.add_relation("Beta", "owns", "Gamma", weight=2.0)

    r = Retriever(passages=list(passages), graph=g).fit()
    question = "What does Acme ultimately produce?"

    # baseline: the Gamma passage shares no terms with the question
    baseline = r.retrieve(question, k=2)
    baseline_texts = [p.text for p in baseline]
    assert passages[1] not in baseline_texts  # answer missed without graph

    # graph-augmented: expansion adds Beta + Gamma terms, surfacing the answer
    augmented, expanded = r.graph_augmented_retrieve(question, k=2, hops=2)
    augmented_texts = [p.text for p in augmented]
    assert "Gamma" in expanded
    assert passages[1] in augmented_texts  # answer retrieved with graph

    # and it ranks at the top
    assert augmented[0].text == passages[1]
