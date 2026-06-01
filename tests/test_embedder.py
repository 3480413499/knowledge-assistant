"""Tests for embedder.py"""
import pytest
import numpy as np
from src.embedder import Embedder


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        """全测试共用 embedder，避免重复加载模型."""
        return Embedder()

    def test_embed_single_text_returns_ndarray(self, embedder):
        """嵌入单条文本应返回 numpy 数组"""
        vec = embedder.embed("你好世界")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == 1024

    def test_embed_batch_returns_2d_array(self, embedder):
        """批量嵌入应返回二维数组"""
        texts = ["第一段", "第二段", "第三段"]
        vecs = embedder.embed_batch(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.ndim == 2
        assert vecs.shape == (3, 1024)

    def test_semantic_similarity(self, embedder):
        """语义相近的句子向量余弦相似度应较高"""
        v1 = embedder.embed("今天天气真好")
        v2 = embedder.embed("天气很不错呢")
        v3 = embedder.embed("Python 编程语言的学习")

        sim_12 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sim_13 = float(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)))

        assert sim_12 > sim_13, (
            f"相似语义应该有更高相似度: sim12={sim_12:.3f}, sim13={sim_13:.3f}"
        )

    def test_empty_text_raises(self, embedder):
        """空字符串应抛出 ValueError"""
        with pytest.raises(ValueError):
            embedder.embed("")

    def test_embed_query_returns_ndarray(self, embedder):
        """查询嵌入（embed_query）应返回一维向量"""
        vec = embedder.embed_query("如何学习深度学习？")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == 1024
