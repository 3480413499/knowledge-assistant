"""Tests for rag_engine.py"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.rag_engine import RAGEngine


class TestRAGEngine:
    @pytest.fixture
    def engine(self):
        with patch("src.rag_engine.Embedder") as mock_embedder_cls, \
             patch("src.rag_engine.LLMClient") as mock_llm_cls:

            mock_embedder = MagicMock()
            mock_embedder.dimension = 1024
            mock_embedder.embed_batch.side_effect = lambda texts: np.random.rand(len(texts), 1024)
            mock_embedder.embed_query.return_value = np.random.rand(1024)
            mock_embedder_cls.return_value = mock_embedder

            mock_llm = MagicMock()
            mock_llm.answer.return_value = "这是基于文档的回答。\n📎 参考来源：[1] test.pdf 第1页 — 「原文」"
            mock_llm_cls.return_value = mock_llm

            engine = RAGEngine(
                llm_api_key="test-key",
                persist_directory=None,
            )
            engine.clear()
            yield engine

    def test_ingest_text_increases_count(self, engine):
        initial = engine.get_document_count()
        engine.ingest_text("测试文档内容。", "test.pdf", 1)
        assert engine.get_document_count() == initial + 1

    def test_ingest_duplicate_is_skipped(self, engine):
        text = "独一无二的内容用于去重测试。"
        engine.ingest_text(text, "test.pdf", 1)
        count = engine.get_document_count()
        engine.ingest_text(text, "test.pdf", 1)
        assert engine.get_document_count() == count

    def test_ingest_file_computes_md5(self, engine, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("测试文件内容", encoding="utf-8")
        engine.ingest_file(f)
        assert engine.is_file_ingested(f) is True

    def test_query_returns_answer(self, engine):
        engine.ingest_text("SLAM 是同步定位与地图构建。", "a.pdf", 1)
        result = engine.query("什么是 SLAM？")
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "sources" in result

    def test_query_empty_store_returns_not_found(self, engine):
        result = engine.query("任意问题")
        assert "暂无" in result["answer"] or "上传" in result["answer"] or "为空" in result["answer"]

    def test_get_sources_list(self, engine):
        engine.ingest_text("内容A", "file_a.pdf", 1)
        engine.ingest_text("内容B", "file_b.pdf", 1)
        sources = engine.get_sources()
        assert "file_a.pdf" in sources
        assert "file_b.pdf" in sources

    def test_clear_knowledge_base(self, engine):
        engine.ingest_text("测试", "a.pdf", 1)
        engine.clear()
        assert engine.get_document_count() == 0
