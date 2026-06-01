"""Tests for vector_store.py"""
import pytest
import numpy as np
from src.vector_store import VectorStore


class TestVectorStore:
    @pytest.fixture
    def store(self):
        """每个测试使用独立的内存存储."""
        import chromadb
        client = chromadb.Client()
        store = VectorStore(collection_name="test_kb")
        store._client = client
        store._collection = client.get_or_create_collection(name="test_kb")
        yield store
        try:
            client.delete_collection("test_kb")
        except Exception:
            pass

    def test_add_and_count(self, store):
        docs = [
            {"content": "文档一：Python 编程", "file": "a.pdf", "page": 1},
            {"content": "文档二：机器学习", "file": "a.pdf", "page": 2},
        ]
        embeddings = [np.random.rand(1024).tolist() for _ in docs]
        store.add(docs, embeddings)
        assert store.count() == 2

    def test_search_returns_top_k(self, store):
        docs = [
            {"content": f"内容{i}", "file": "a.pdf", "page": i}
            for i in range(1, 11)
        ]
        embeddings = [np.random.rand(1024).tolist() for _ in docs]
        store.add(docs, embeddings)

        query_vec = np.random.rand(1024).tolist()
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3
        for r in results:
            assert "content" in r
            assert "file" in r
            assert "page" in r

    def test_search_empty_store_returns_empty(self, store):
        results = store.search(np.random.rand(1024).tolist(), top_k=4)
        assert results == []

    def test_clear_removes_all(self, store):
        docs = [{"content": "测试", "file": "a.pdf", "page": 1}]
        store.add(docs, [np.random.rand(1024).tolist()])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_has_content(self, store):
        docs = [{"content": "独一无二的内容", "file": "a.pdf", "page": 1}]
        store.add(docs, [np.random.rand(1024).tolist()])
        assert store.has_content("独一无二的内容") is True
        assert store.has_content("不存在的内容") is False

    def test_get_sources(self, store):
        docs = [
            {"content": "a", "file": "file1.pdf", "page": 1},
            {"content": "b", "file": "file1.pdf", "page": 2},
            {"content": "c", "file": "file2.pdf", "page": 1},
        ]
        store.add(docs, [np.random.rand(1024).tolist() for _ in docs])
        sources = store.get_sources()
        assert "file1.pdf" in sources
        assert "file2.pdf" in sources
        assert len(sources) == 2
