"""Offline equivalence and encoder-work checks for shared embedding caches."""
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from llm import ModelError
from memory import Document
from retrieve import Embedder, Retriever


class Tokenizer:
    model_max_length = 64

    def __init__(self):
        self.calls = 0

    def encode(self, text, **kwargs):
        self.calls += 1
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class Encoder:
    max_seq_length = 64

    def __init__(self, model, device=None):
        self.model, self.tokenizer, self.calls = model, Tokenizer(), []

    def encode(self, values, **kwargs):
        self.calls.append(list(values))
        return np.asarray([
            list(hashlib.sha256((self.model + text).encode()).digest()[:8])
            for text in values
        ], dtype=np.float32)


@pytest.fixture
def make_embedder(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setenv("EMBED_CACHE_ITEMS", "4096")
    monkeypatch.setenv("EMBED_QUERY_PREFIX", "")
    monkeypatch.setenv("EMBED_DOCUMENT_PREFIX", "")
    return lambda model="offline": Embedder("local", model)


def doc(key, text, date=""):
    return Document(key, text, date, [], [], [])


def test_changed_memories_only_encode_changed_chunks_and_share_queries(make_embedder, tmp_path):
    cached, uncached = make_embedder(), make_embedder()
    uncached.cache_items = 0
    base = [doc(str(i), f"memory entry {i}") for i in range(20)]
    for sample in range(8):
        documents = base + [doc("new", f"new candidate {sample}")]
        fast = Retriever(documents, cached, tmp_path / "cached")
        reference = Retriever(documents, uncached, tmp_path / "uncached")
        np.testing.assert_allclose(fast.vectors, reference.vectors)
        assert fast.search("which memory entry", 5) == reference.search("which memory entry", 5)
    # 20 shared entries + 8 candidate entries + one shared query, versus
    # 8 * (21 document entries + one query) in separate uncached indexes.
    assert sum(map(len, cached.encoder.calls)) == 29
    assert sum(map(len, uncached.encoder.calls)) == 176


def test_model_prefix_role_and_length_budget_do_not_share_vectors(make_embedder):
    first, second = make_embedder("first"), make_embedder("second")
    original = first.encode(["shared"])
    assert not np.allclose(original, second.encode(["shared"]))
    first.encode(["shared"], query=True)
    assert len(first.encoder.calls) == 2  # query/document roles stay distinct.
    first.query_prefix = "query: "
    prefixed = first.encode(["shared"], query=True)
    assert first.encoder.calls[-1] == ["query: shared"]
    assert not np.allclose(prefixed, original)
    long = "x" * 100
    before = first.encode([long])
    first.encoder.max_seq_length = 32
    after = first.encode([long])
    assert not np.allclose(before, after)
    assert len(first.encoder.calls[-1][0]) == 30


def test_split_cache_avoids_tokenization_and_returns_independent_lists(make_embedder):
    embedder = make_embedder()
    text = "x" * 100
    first = embedder.split(text)
    calls = embedder.encoder.tokenizer.calls
    first[0] = "mutated by caller"
    assert embedder.split(text)[0] != first[0]
    assert embedder.encoder.tokenizer.calls == calls
    embedder.encoder.max_seq_length = 32
    assert len(embedder.split(text)) > len(first)
    assert embedder.encoder.tokenizer.model_max_length == 64


def test_bounded_lru_handles_large_batches_duplicates_and_caller_mutation(make_embedder):
    embedder = make_embedder()
    embedder.cache_items = 2
    values = embedder.encode(["a", "b", "a", "c"])
    assert embedder.encoder.calls == [["a", "b", "c"]]
    assert len(embedder._vector_cache) == 2
    np.testing.assert_array_equal(values[0], values[2])
    c = values[-1].copy()
    values[-1] = 0
    np.testing.assert_array_equal(embedder.encode(["c"])[0], c)
    assert len(embedder.encoder.calls) == 1
    embedder.encode(["a"])
    assert embedder.encoder.calls[-1] == ["a"]
    for text in ["a", "b", "c"]:
        embedder.split(text)
    assert len(embedder._split_cache) == 2


def test_bad_vectors_are_never_cached(make_embedder, monkeypatch):
    embedder = make_embedder()
    original = embedder.encoder.encode
    monkeypatch.setattr(embedder.encoder, "encode", lambda *a, **kw: [[float("nan")]])
    with pytest.raises(ModelError):
        embedder.encode(["retry me"])
    assert not embedder._vector_cache
    monkeypatch.setattr(embedder.encoder, "encode", original)
    assert np.isfinite(embedder.encode(["retry me"])).all()


def test_existing_disk_index_is_still_reused(make_embedder, tmp_path):
    documents = [doc("a", "alpha", "2020-01-01"), doc("b", "beta")]
    first = Retriever(documents, make_embedder(), tmp_path)
    fresh = make_embedder()
    loaded = Retriever(documents, fresh, tmp_path)
    assert fresh.encoder.calls == []
    np.testing.assert_array_equal(first.vectors, loaded.vectors)
    assert first.search("alpha", 2) == loaded.search("alpha", 2)
    changed_date = [doc("a", "alpha", "2021-01-01"), documents[1]]
    Retriever(changed_date, fresh, tmp_path)
    assert any("2021-01-01" in text for batch in fresh.encoder.calls for text in batch)


def test_api_cache_keeps_batch_order_and_invalidates_changed_split_size(monkeypatch):
    from retrieve import Client
    monkeypatch.setenv("EMBED_CACHE_ITEMS", "16")
    monkeypatch.setenv("EMBED_CHUNK_CHARS", "32")
    monkeypatch.setenv("EMBED_QUERY_PREFIX", "q:")
    monkeypatch.setenv("EMBED_DOCUMENT_PREFIX", "d:")
    calls = []

    def embed(client, values):
        calls.append(values)
        return [[len(text), sum(map(ord, text))] for text in values]

    monkeypatch.setattr(Client, "embed", embed)
    model = Embedder("api", "offline-api")
    result = model.encode(["a", "b", "a"])
    assert calls == [["d:a", "d:b"]]
    np.testing.assert_array_equal(result[0], result[2])
    model.encode(["b", "c"])
    assert calls[-1] == ["d:c"]
    model.encode(["a"], query=True)
    assert calls[-1] == ["q:a"]
    assert len(model.split("x" * 80)) == 3
    monkeypatch.setenv("EMBED_CHUNK_CHARS", "64")
    assert len(model.split("x" * 80)) == 2


def test_cache_can_be_disabled_or_reject_invalid_limit(make_embedder, monkeypatch):
    monkeypatch.setenv("EMBED_CACHE_ITEMS", "0")
    embedder = make_embedder()
    embedder.encode(["same", "same"])
    embedder.encode(["same"])
    assert embedder.encoder.calls == [["same", "same"], ["same"]]
    assert not embedder._vector_cache
    monkeypatch.setenv("EMBED_CACHE_ITEMS", "-1")
    with pytest.raises(ValueError, match="EMBED_CACHE_ITEMS"):
        make_embedder()


def test_search_many_batches_queries_without_changing_fusion(make_embedder, tmp_path, monkeypatch):
    documents = [doc(str(i), f"memory entry {i}") for i in range(10)]
    fast_encoder, reference_encoder = make_embedder(), make_embedder()
    fast = Retriever(documents, fast_encoder, tmp_path / "fast")
    reference = Retriever(documents, reference_encoder, tmp_path / "reference")
    monkeypatch.setattr(reference, "_warm_queries", lambda queries: None)
    fast_encoder.encoder.calls.clear()
    reference_encoder.encoder.calls.clear()
    queries = ["memory entry " + str(i) for i in range(35)] + ["", "   ", "memory entry 0"]
    assert fast.search_many(queries, 4) == reference.search_many(queries, 4)
    assert [len(batch) for batch in fast_encoder.encoder.calls] == [32, 3]
    assert len(reference_encoder.encoder.calls) == 35
    fast.search_many(queries, 4)
    assert len(fast_encoder.encoder.calls) == 2


def test_search_many_empty_invalid_and_bm25_only_do_not_encode(make_embedder, tmp_path):
    documents = [doc("a", "alpha")]
    embedder = make_embedder()
    index = Retriever(documents, embedder, tmp_path)
    embedder.encoder.calls.clear()
    assert index.search_many(["", "   "]) == []
    with pytest.raises(ValueError):
        index.search_many(["alpha"], candidate_k=0)
    with pytest.raises(ValueError):
        index.search_many(["alpha"], expand=-1)
    with pytest.raises(TypeError):
        index.search_many(["alpha"], unsupported=True)
    assert embedder.encoder.calls == []
    plain = Retriever(documents)
    assert plain.search_many(["alpha"]) == plain.search("alpha")
