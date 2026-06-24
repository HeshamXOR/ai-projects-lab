"""Proofs for the inverted index, BM25F, and call graph."""

from core.inverted_index import InvertedIndex
from core.bm25f import BM25F, tokenize
from core.callgraph import CallGraph


def test_inverted_index_boolean_and():
    idx = InvertedIndex()
    idx.add(0, "read the file from disk")
    idx.add(1, "write the file to disk")
    idx.add(2, "compute a checksum")
    assert idx.boolean_and("file disk") == {0, 1}
    assert idx.boolean_and("checksum") == {2}


def test_inverted_index_phrase():
    idx = InvertedIndex()
    idx.add(0, "read the file from disk")
    idx.add(1, "the file read was corrupted")
    # "read the file" is an adjacent phrase only in doc 0
    assert idx.phrase_search("read the file") == {0}


def test_bm25f_field_weighting():
    # query "read" matches doc A's NAME and doc B's BODY; name is weighted higher
    docs = [
        {"name": "read_file", "signature": "read_file(path)", "comments": "", "body": "return open(path).read()"},
        {"name": "helper", "signature": "helper()", "comments": "", "body": "we read data here read read"},
    ]
    bm = BM25F()
    bm.index(docs)
    res = bm.search("read", k=2)
    assert res[0][0] == 0  # the function literally named read_file wins


def test_camel_snake_tokenization():
    assert "read" in tokenize("readFile")
    assert "file" in tokenize("readFile")
    assert "read" in tokenize("read_file")


def test_callgraph_resolves_relationships():
    src = '''
def a():
    b()
    c()

def b():
    c()

def c():
    pass
'''
    g = CallGraph()
    g.add_source(src)
    assert g.callees("a") == {"b", "c"}
    assert g.callers("c") == {"a", "b"}
    assert "b" in g.related("a")
