"""RAG 引擎 — 核心编排层，连接嵌入、向量库、LLM."""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import TOP_K, DEFAULT_KB_NAME, kb_to_collection, load_kb_names
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
        embedder: Optional[Embedder] = None,
        kb_name: str = DEFAULT_KB_NAME,
    ):
        self._embedder = embedder or Embedder()
        self._kb_display_name = kb_name
        self._vector_store = VectorStore(
            collection_name=kb_to_collection(kb_name),
            persist_directory=persist_directory,
        )
        self._llm = LLMClient(
            api_key=llm_api_key,
            base_url=llm_base_url or "https://api.deepseek.com",
        )
        self._ingested_files: Dict[str, Dict[str, str]] = {kb_name: {}}

    @property
    def kb_name(self) -> str:
        return self._kb_display_name

    def switch_kb(self, name: str) -> None:
        """切换到指定知识库，保留之前的摄入记录."""
        self._kb_display_name = name
        self._vector_store.switch_collection(kb_to_collection(name))
        if name not in self._ingested_files:
            self._ingested_files[name] = {}

    def list_kb(self) -> list:
        """列出所有知识库名称（显示名）."""
        return load_kb_names()

    def delete_kb(self, name: str) -> None:
        """删除指定知识库."""
        self._vector_store.delete_collection(kb_to_collection(name))
        self._ingested_files.pop(name, None)

    def _current_ingested(self) -> Dict[str, str]:
        """获取当前 KB 的摄入记录."""
        name = self._kb_display_name
        if name not in self._ingested_files:
            self._ingested_files[name] = {}
        return self._ingested_files[name]

    _INGEST_BATCH = 500  # 每批嵌入 500 条，避免 GPU 显存峰值

    def ingest_file(self, file_path: Path, progress_callback=None) -> int:
        """摄入单个文件：解析 → 分块 → 嵌入 → 存储。返回新增块数."""
        md5 = FileParser.compute_md5(file_path)
        key = str(file_path.absolute())
        ingested = self._current_ingested()

        if key in ingested and ingested[key] == md5:
            return 0

        text = FileParser.parse(file_path)
        if not text.strip():
            return 0

        chunks = self._ingest_text(text, file_path.name, progress_callback=progress_callback)
        ingested[key] = md5
        return len(chunks)

    def ingest_text(self, text: str, file_name: str, page_num: int = 1,
                    progress_callback=None) -> int:
        """摄入纯文本（测试用），返回块数."""
        chunks = self._ingest_text(text, file_name, page_num,
                                   progress_callback=progress_callback)
        return len(chunks)

    def _ingest_text(
        self, text: str, file_name: str, page_num: int = 1,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """文本摄入内部实现."""
        import hashlib

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

        # 哈希去重：用已有文本的 MD5 前 16 位做内存 set 比较
        existing_hashes = self._vector_store.get_text_hashes()
        new_chunks = []
        for c in all_chunks:
            h = hashlib.md5(c["content"].encode()).hexdigest()[:16]
            if h not in existing_hashes:
                new_chunks.append(c)
                existing_hashes.add(h)

        total = len(new_chunks)
        if progress_callback:
            progress_callback(0, total, "开始向量化...")

        # 分批嵌入 + 分批存储，避免 GPU 显存峰值 & 支持断点恢复
        for i in range(0, total, self._INGEST_BATCH):
            batch = new_chunks[i:i + self._INGEST_BATCH]
            embeddings = self._embedder.embed_batch(
                [c["content"] for c in batch]
            )
            self._vector_store.add(batch, embeddings.tolist())
            if progress_callback:
                done = min(i + self._INGEST_BATCH, total)
                progress_callback(done, total, f"向量化中 {done}/{total}")

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
        """清空当前知识库."""
        self._vector_store.clear()
        self._current_ingested().clear()
