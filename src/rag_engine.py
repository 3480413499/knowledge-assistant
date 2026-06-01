"""RAG 引擎 — 核心编排层，连接嵌入、向量库、LLM."""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import TOP_K
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.file_parser import FileParser
from src.text_splitter import TextSplitter
from src.llm_client import LLMClient


class RAGEngine:
    """RAG 核心引擎，协调文档摄入和问答流程."""

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self._embedder = Embedder()
        self._vector_store = VectorStore(persist_directory=persist_directory)
        self._llm = LLMClient(
            api_key=llm_api_key,
            base_url=llm_base_url or "https://api.deepseek.com",
        )
        self._ingested_files: Dict[str, str] = {}

    def ingest_file(self, file_path: Path) -> int:
        """摄入单个文件：解析 → 分块 → 嵌入 → 存储。返回新增块数."""
        md5 = FileParser.compute_md5(file_path)
        key = str(file_path.absolute())

        if key in self._ingested_files and self._ingested_files[key] == md5:
            return 0

        text = FileParser.parse(file_path)
        if not text.strip():
            return 0

        chunks = self._ingest_text(text, file_path.name)
        self._ingested_files[key] = md5
        return len(chunks)

    def ingest_text(self, text: str, file_name: str, page_num: int = 1) -> int:
        """摄入纯文本（测试用），返回块数."""
        chunks = self._ingest_text(text, file_name, page_num)
        return len(chunks)

    def _ingest_text(
        self, text: str, file_name: str, page_num: int = 1
    ) -> List[Dict[str, Any]]:
        """文本摄入内部实现."""
        parts = text.split("\n\n[第")
        all_chunks = []

        if len(parts) > 1:
            for part in parts:
                if not part.strip():
                    continue
                try:
                    page_str, content = part.split("页]\n", 1)
                    page = int(page_str)
                except (ValueError, IndexError):
                    content = part
                    page = page_num
                chunks = TextSplitter.split_with_metadata(
                    content, file_name, page
                )
                all_chunks.extend(chunks)
        else:
            chunks = TextSplitter.split_with_metadata(
                text, file_name, page_num
            )
            all_chunks.extend(chunks)

        new_chunks = []
        for chunk in all_chunks:
            if not self._vector_store.has_content(chunk["content"]):
                new_chunks.append(chunk)

        if new_chunks:
            embeddings = self._embedder.embed_batch(
                [c["content"] for c in new_chunks]
            )
            self._vector_store.add(new_chunks, embeddings.tolist())

        return new_chunks

    def query(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """用户提问，返回回答 + 来源."""
        if self._vector_store.count() == 0:
            return {
                "answer": "📭 知识库为空，请先在侧边栏上传文档再提问哦～",
                "sources": [],
            }

        query_vec = self._embedder.embed_query(question)
        retrieved = self._vector_store.search(query_vec.tolist(), top_k=top_k)
        answer = self._llm.answer(question, retrieved)

        return {
            "answer": answer,
            "sources": [
                {
                    "file": r["file"],
                    "page": r["page"],
                    "snippet": r["content"][:100],
                }
                for r in retrieved
            ],
        }

    def is_file_ingested(self, file_path: Path) -> bool:
        """检查文件是否已摄入."""
        return str(file_path.absolute()) in self._ingested_files

    def get_document_count(self) -> int:
        """返回知识库文档块总数."""
        return self._vector_store.count()

    def get_sources(self) -> List[str]:
        """返回所有已摄入的源文件名."""
        return self._vector_store.get_sources()

    def clear(self) -> None:
        """清空知识库."""
        self._vector_store.clear()
        self._ingested_files.clear()
