"""嵌入模型模块 — BGE-M3 本地嵌入，文本转向量."""
import numpy as np
from typing import List
from src.config import LOCAL_EMBEDDING_PATH, EMBEDDING_MODEL_NAME


class Embedder:
    """文本嵌入器，使用 BGE-M3 模型（懒加载，优先本地路径）."""

    def __init__(self, model_path: str = None, device: str = None):
        # 优先本地路径 -> 回退模型名（会触发 HF 下载，国内可能超时）
        self._model_path = model_path or LOCAL_EMBEDDING_PATH or EMBEDDING_MODEL_NAME
        self._model = None
        self._dim = None
        # 自动检测设备：GPU 可用就用 cuda，否则 CPU
        if device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _ensure_loaded(self):
        """首次使用时加载模型."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_path, device=self._device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """嵌入向量维度."""
        self._ensure_loaded()
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """将单条文本转为向量."""
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        self._ensure_loaded()
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入，返回 shape=(N, dim) 的数组."""
        if not texts:
            raise ValueError("文本列表不能为空")
        self._ensure_loaded()
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入查询文本."""
        return self.embed(query)
