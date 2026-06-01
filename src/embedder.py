"""嵌入模型模块 — BGE-M3 本地嵌入，文本转向量."""
import numpy as np
from typing import List
from src.config import EMBEDDING_MODEL_NAME


class Embedder:
    """文本嵌入器，使用 BGE-M3 模型."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """嵌入向量维度."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """将单条文本转为向量."""
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入，返回 shape=(N, dim) 的数组."""
        if not texts:
            raise ValueError("文本列表不能为空")
        return self._model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询文本."""
        return self.embed(query)
